import os
import io
import json
import torch
import time
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
from model_bc import BinaryClassifier
from multitask import MultiTask
from yelp import Yelp


from utils import idx2word
import argparse

def main(args):
 
    # create dir name
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    ts = ts.replace(':', '-')
    ts = ts+'-'+args.dataset
        
    if(args.attention):
        ts = ts+'-self-attn'

    ts = ts + "-" + str(args.epochs)

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

    # print(type(int(datasets['train'].yelp_max_sequence_length)))
    
        
    
    max_sequence_length = datasets['train'].max_sequence_length

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
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        attention=args.attention,
        dataset=args.dataset,
        
    )

    # init model object
    model = BinaryClassifier(**params)

    if torch.cuda.is_available():
        model = model.cuda()

    # logging
    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args, ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    # make dir
    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    # write params to json and save
    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)    


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0

    overall_losses = defaultdict(dict)
    loss_at_epoch = {
        'loss': 0.0,
        'acc': 0.0
    }

    for epoch in range(args.epochs):

        # do train and then test
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
                preds = model(batch['input'], batch['length'])

                # ae loss calculation
                loss = nn.BCELoss()(preds, batch['label'].type(torch.FloatTensor).cuda())
        
                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()  # flush grads            
                    loss.backward()  
                    optimizer.step()
                    step += 1

                # calculate accruracies
                preds = torch.argmax(preds, dim = 1)
                ground_truth = torch.argmax(batch['label'], dim = 1)
                
                acc = (preds==ground_truth).sum()/batch_size
               
                # try sample to verify style classifier is working
                # print(idx2word(batch['target'][0:1], i2w=i2w, pad_idx=w2i['<pad>']))
                # print(batch['label'][0])
                # print("neg: {}, pos: {}".format(style_preds[0:1,0], style_preds[0:1,1]))

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("-----------------------------------------------------------------------")
                    print("%s Batch %04d/%i, Loss %9.4f, Acc %9.4f" % (split.upper(), iteration, len(data_loader)-1, loss.item(), acc))

            # save checkpoint
            if split == 'train':
                loss_at_epoch['loss'] = float(loss)
                loss_at_epoch['acc'] = float(acc)
                overall_losses[len(overall_losses)] = loss_at_epoch
                checkpoint_path = os.path.join(args.save_model_path, "E%i.pytorch" % epoch)
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)
            
    # write losses to json
    with open(os.path.join(args.save_model_path, 'losses.json'), 'w') as f:
        json.dump(overall_losses, f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--min_occ', type=int, default=2)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32) #works well at 8 for some reason
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-save', '--save_model_path', type=str, default='./saved_text_classifiers/')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true') 
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-attention', '--attention', type=bool, default=False)
    
    parser.add_argument('-dataset', '--dataset', type=str, default='yelp', required=True)

    args = parser.parse_args()
    
    #NOTE: if bidirection = true, NLL will overfit, Classifiers will underfit, so dont use this
    # args.bidirectional = True

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']

    main(args)