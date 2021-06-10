import os
import json
import torch
import argparse
from multiprocessing import cpu_count
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from torch.utils.data import DataLoader
from model_rep import SentenceVae
from utils import to_var, idx2word, interpolate, load_model_params_from_checkpoint
from dataset_preproc_scripts.yelp import Yelp
# from snli import SNLI
# from multitask import MultiTask
from random import randint
import numpy as np

def main(args):

    # load checkpoint
    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    saved_dir_name = args.load_checkpoint.split('/')[2]
    
    # load params
    params_path = './saved_vae_models/'+saved_dir_name+'/model_params.json'
    if not os.path.exists(params_path):
        raise FileNotFoundError(params_path)

    params = load_model_params_from_checkpoint(params_path)

    # set model and dataset according to options
    
    with open('./data/yelp/yelp.vocab.json', 'r') as file:
        vocab = json.load(file)
    vaemodel = SentenceVae
    dataset = Yelp
    
    # load dataset
    split = 'train'
    datasets = defaultdict(dict)
    datasets[split] = dataset(split=split, create_data=False, min_occ=2)
    w2i, i2w = vocab['w2i'], vocab['i2w']

    # create model
    model = vaemodel(**params)
    print(model)
    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    print("Computing mean style vectors...")

    ##################### get mean style_z ###################
    # create dataloader
    data_loader = DataLoader(
        dataset=datasets[split],
        batch_size=args.batch_size,
        shuffle=split == 'train',
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available()
    )

    for iteration, batch in enumerate(data_loader):

        # get batch size
        batch_size = batch['input'].size(0)

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = to_var(v)
        
        style_z, content_z = model.get_style_content_space(batch['input'])
        batch_labels = batch['label']
        preds = torch.argmax(batch_labels, dim = 1)
        
        if(iteration==0):
            pos_preds = style_z[preds == 1].sum(axis = 0) #filter out preds and accumulate
            neg_preds = style_z[preds == 0].sum(axis = 0)
        else:
            pos_preds += style_z[preds == 1].sum(axis = 0)
            neg_preds += style_z[preds == 0].sum(axis = 0)
        
        # if iteration==1000:
        #     # get average
        #     mean_pos_style = pos_preds/((iteration+1)*args.batch_size)
        #     mean_neg_style = neg_preds/((iteration+1)*args.batch_size)
        #     break

    mean_pos_style = pos_preds/((iteration+1)*args.batch_size)
    mean_neg_style = neg_preds/((iteration+1)*args.batch_size)
            
    ################ Experiment 1: flip sentiment of sentences #############3############
    for i in range(100):

        #pick a sentence
        sent1 = datasets[split].__getitem__(i)
        
        # get the lspace vectors for sent1 and sent2
        sent1_tokens = torch.tensor(sent1['input']).unsqueeze(0)
        batch = torch.cat((sent1_tokens, sent1_tokens), 0).cuda()

        style_z, content_z = model.encode_to_lspace(batch, zero_noise=True)

        # print outputs
        print("{}).---------------------------------------------------".format(i))
        print("Sentence 1 with label: {}".format(sent1['label']))
        print(*idx2word(sent1_tokens, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
        print()

        label = np.argmax(sent1['label'])
        if(label == 1): #if sentiment is positive flip to negative
            print("Converting from positive to negative:")
            final_z = torch.cat((mean_neg_style, content_z[0]), -1).unsqueeze(0)
        else: # and vice versa
            print("Converting from negative to positive:")
            final_z = torch.cat((mean_pos_style, content_z[0]), -1).unsqueeze(0)

        words = model.final_z_to_words(sent1['input'] ,final_z)
        # samples, _ = model.inference(z=final_z)
        print(*idx2word(words, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-p', '--load_params', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32) 

    args = parser.parse_args()
    main(args)
