import os
import io
import json
import torch
import time
import math
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from multiprocessing import cpu_count
from utils import to_var, idx2word, expierment_name
from torch.utils.data import DataLoader
from nltk.tokenize import TweetTokenizer
from collections import OrderedDict, defaultdict
from utils import OrderedCounter
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from model_rep import SentenceVae
from dataset_preproc_scripts.multitask import MultiTask
from dataset_preproc_scripts.yelp import Yelp
from dataset_preproc_scripts.snli import SNLI
from utils import idx2word
import argparse

def get_entropy_loss(preds, epsilon=1e-8):
    entropy = torch.sum(-preds * torch.log(preds+epsilon), dim=1)
    return torch.mean(entropy)

def get_anneal_weight(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)
    elif anneal_function == 'sigmoid':
        kl_tot_iterations = 20000
        return (math.tanh((step - kl_tot_iterations * 1.5)/(kl_tot_iterations / 3))+ 1)
    else:
        print("something wrong in KL annealing")
        exit()

def get_kl_loss(mean, logv):
    '''Return KL loss between P(z|X) and a standard gaussian dist.'''
    KL_loss = torch.mean(-0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp(), dim=1))
    return KL_loss

def main(args):
 
    # create dir name
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    ts = ts.replace(':', '-')
    ts = ts+'-'+args.dataset+'-'
    
    if(args.w2v):
        ts = ts+'-w2v-'

    ts = ts + str(args.epochs)

    if(args.dataset == "multitask"):
        print("Running multitask dataset!")
        dataset = MultiTask
    if(args.dataset == "snli"):
        print("Running SNLI!")
        dataset = SNLI
    if(args.dataset == "yelp"):
        print("Running Yelp!")
        dataset = Yelp

    # prepare dataset
    splits = ['train', 'test']

    # create dataset object
    datasets = OrderedDict()

    # create test and train split in data, also preprocess
    for split in splits:
        print("creating dataset for: {}".format(split))
        datasets[split] = dataset(
            split=split,
            create_data=args.create_data,
            min_occ=args.min_occ
        )

    i2w = datasets['train'].get_i2w()
    w2i = datasets['train'].get_w2i()

    if(args.dataset == "multitask"):
        max_sequence_length = max(datasets['train'].yelp_max_sequence_length, datasets['train'].snli_max_sequence_length)
    else:
        max_sequence_length = datasets['train'].max_sequence_length

    #get w2v pretrained embeds matrix for dataset
    if(args.w2v):
        path_to_w2v_weights = "./data/{}/{}_w2v_weights.npy".format(args.dataset, args.dataset)
    else:
        path_to_w2v_weights = None 

    # get training params
    params = dict(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        max_sequence_length=max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dataset=args.dataset,
        content_bow_dim=datasets['train'].bow_hidden_dim,
        path_to_w2v_weights=path_to_w2v_weights
    )

    # init model object
    model = SentenceVae(**params)

    if torch.cuda.is_available():
        model = model.cuda()

    # make dir
    save_model_path = os.path.join(datasets["train"].save_model_path, ts)
    os.makedirs(save_model_path)

    # write params to json and save
    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)    

    # defining NLL loss to measure accuracy of the decoding
    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='mean')

    def get_nll_loss(logp, target, length):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :datasets["train"].max_sequence_length].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        NLL_loss = NLL(logp, target)

        return NLL_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0

    loss_at_epoch = {
        'nll_loss': 0.0,
        'kl_loss': 0.0,
        'style_mul_loss': 0.0,
        'content_mul_loss': 0.0,   
        'style_disc_loss' : 0.0,
        'content_disc_loss' : 0.0,
        'style_adv_entropy' : 0.0,
        'content_adv_entropy' : 0.0,
        'style_mul_acc' : 0.0,
        'content_adv_acc' : 0.0
    }

    path_to_logs = "./saved_vae_models/" + ts + "/logs.txt"
    logs_file = open(path_to_logs, 'w')

    for epoch in range(args.epochs):

        for split in splits:

            # create dataloader
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split == 'train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            # tracker used to track the loss
            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            # start batch wise training/testing
            for iteration, batch in enumerate(data_loader):

                # get batch size
                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

               
                # Forward pass
                logp, style_mean, style_logv, content_mean, content_logv, style_preds, content_preds, style_adv_preds, content_adv_preds = model(batch['input'], batch['length'], batch['label'], batch['bow'])

                #NLL recon loss calculation
                NLL_loss = get_nll_loss(logp, batch['target'], batch['length'])

                #KLL loss calculation
                KL_weight = get_anneal_weight(args.anneal_function, step, args.k, args.x0)
                style_KL_loss = get_kl_loss(style_mean, style_logv)
                content_KL_loss = get_kl_loss(content_mean, content_logv)
                
                # multi task loss calculation
                style_loss = nn.BCELoss()(style_preds, batch['label'].type(torch.FloatTensor).cuda()) 
                content_loss = nn.BCELoss()(content_preds, batch['bow'].type(torch.FloatTensor).cuda())

                # entropy loss calculation
                style_adv_loss = get_entropy_loss(style_adv_preds)
                content_adv_loss = get_entropy_loss(content_adv_preds)

                # adv discriminator losses
                content_disc_loss = nn.BCELoss()(content_adv_preds, batch['label'].type(torch.FloatTensor).cuda())
                style_disc_loss = nn.BCELoss()(style_adv_preds, batch['bow'].type(torch.FloatTensor).cuda())

                KL_loss = style_KL_loss + content_KL_loss
                
                # total loss calculation
                if(not args.disentangle):
                    loss = NLL_loss +  0.03*KL_weight*KL_loss + 10*style_loss + 3*content_loss - 1*style_adv_loss - 0.03*content_adv_loss
                else:
                    loss = NLL_loss + KL_weight * KL_loss
                    print("not disentangling")
                    exit(0)
                    
                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()  # flush grads
                    content_disc_loss.backward(retain_graph=True)
                    style_disc_loss.backward(retain_graph=True)
                    loss.backward()  
                    optimizer.step()
                    step += 1

                # calculate accruracies
                style_multi_preds = torch.argmax(style_preds, dim = 1)
                content_adv_preds = torch.argmax(content_adv_preds, dim = 1)
                ground_truth = torch.argmax(batch['label'], dim = 1)
                
                style_multi_acc = (style_multi_preds==ground_truth).sum()/batch_size
                content_adv_acc = (content_adv_preds==ground_truth).sum()/batch_size
                
                # try sample to verify style classifier is working
                # print(idx2word(batch['target'][0:1], i2w=i2w, pad_idx=w2i['<pad>']))
                # print(batch['label'][0])
                # print("neg: {}, pos: {}".format(style_preds[0:1,0], style_preds[0:1,1]))

                # bookkeeping
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.view(1, -1)), dim=0)

                      
                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("-----------------------------------------------------------------------")
                    print("Epoch: %i %s Batch %04d/%i\n, Loss %9.4f\n, NLL-Loss %9.4f \n, KL-Loss %9.4f\n, KL-Weight %6.3f\n, Style-Mul-Loss %9.4f\n, Content-Mul-Loss %9.4f\n, Style-Adv-Entropy %9.4f\n, Content-Adv-Entropy %9.4f\n, Content-disc-loss %9.4f\n, Style-disc-loss %9.4f\n, Style Multi Acc %9.4f\n, Content Disc Acc %9.4f\n"
                          % (epoch, split.upper(), iteration, len(data_loader)-1, loss.item(), NLL_loss.item()/batch_size,
                             KL_loss.item()/batch_size, KL_weight, style_loss, content_loss, style_adv_loss, content_adv_loss, content_disc_loss , style_disc_loss, style_multi_acc, content_adv_acc))
    
            print("%s Epoch %02d/%i, Mean ELBO %9.4f" %(split.upper(), epoch, args.epochs, tracker['ELBO'].mean()))

            # save checkpoint if train else print logs
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)
            else:
                loss_at_epoch['nll_loss'] = float(NLL_loss/args.batch_size)
                loss_at_epoch['kl_loss'] = float(KL_loss)
                loss_at_epoch['style_mul_loss'] = float(style_loss)
                loss_at_epoch['content_mul_loss'] = float(content_loss)
                loss_at_epoch['style_disc_loss'] = float(style_disc_loss.item()),
                loss_at_epoch['content_disc_loss'] = float(content_disc_loss.item()),
                loss_at_epoch['style_adv_entropy'] = float(style_adv_loss.item()),
                loss_at_epoch['content_adv_entropy'] = float(content_adv_loss),
                loss_at_epoch['style_mul_acc'] = float(style_multi_acc),
                loss_at_epoch['content_adv_acc'] = float(content_adv_acc)
                logs_file.write("TEST, EPOCH: {} \n".format(epoch))
                
                for key, value in loss_at_epoch.items(): 
                    logs_file.write('\t%s : %s\n' % (key, value))
            
    logs_file.close()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--min_occ', type=int, default=3)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-ep', '--epochs', type=int, default=3)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-dis', '--disentangle', action='store_true')
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true') 
    parser.add_argument('-ls', '--latent_size', type=int, default=128)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.8)
    parser.add_argument('-af', '--anneal_function', type=str, default='sigmoid')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=20000)
    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-w2v', '--w2v', action='store_true')
    parser.add_argument('-dataset', '--dataset', type=str, default='snli')

    args = parser.parse_args()
    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()
    main(args)