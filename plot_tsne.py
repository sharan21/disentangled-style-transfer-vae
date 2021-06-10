
  
import os
import json
import torch
import argparse
from torch.utils.data import Dataset
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model_rep import SentenceVae
from dataset_preproc_scripts.yelp import Yelp

from utils import to_var, idx2word, interpolate, load_model_params_from_checkpoint


def main(args):

    if not os.path.exists(args.load_checkpoint):    
        raise FileNotFoundError(args.load_checkpoint)

    saved_dir_name = args.load_checkpoint.split('/')[2]
    params_path = './saved_vae_models/'+saved_dir_name+'/model_params.json'
    

    if not os.path.exists(params_path):
        raise FileNotFoundError(params_path)

    # load params
    params = load_model_params_from_checkpoint(params_path)

    # create model
    model = SentenceVae(**params)

    print(model)
    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    datasets = OrderedDict()
    style_tsne_values = np.empty((0, model.style_space_size), int)
    content_tsne_values = np.empty((0, model.content_space_size), int)

    if(model.dataset == "yelp"):
        output_size = 2
        print("Using Yelp!")
        tsne_labels = np.empty((0, output_size), int)
        dataset = Yelp
   

    splits = ['train']

    for split in splits:
        print("creating dataset for: {}".format(split))
        datasets[split] = dataset(
            split=split,
        )

    for split in splits:

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
                batch_labels = batch['label'].cpu().detach().numpy()
                style_z = style_z.cpu().detach().numpy()
                content_z = content_z.cpu().detach().numpy()
                style_tsne_values = np.append(style_tsne_values, style_z, axis=0)
                content_tsne_values = np.append(content_tsne_values, content_z, axis=0)
                tsne_labels = np.append(tsne_labels, batch_labels, axis=0)
            
                if iteration==1000:
                    break
            
    pca = PCA(n_components = 8)
    tsne = TSNE(n_components=2, verbose = 1)
    
    style_pca_result = pca.fit_transform(style_tsne_values)
    content_pca_result = pca.fit_transform(content_tsne_values)
    
    style_tsne_results = tsne.fit_transform(style_pca_result[:])
    content_tsne_results = tsne.fit_transform(content_pca_result[:])
    
    
    #plot style
    color_map = np.argmax(tsne_labels, axis=1)
    plt.figure(figsize=(10,10))
    
    for cl in range(output_size):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(style_tsne_results[indices,0], style_tsne_results[indices, 1], label=cl)

    plt.legend()
    plt.savefig('./experiments/'+model.dataset+'-style-tsne.png')

     #plot content
    color_map = np.argmax(tsne_labels, axis=1)
    plt.figure(figsize=(10,10))
    
    for cl in range(output_size):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(content_tsne_results[indices,0], content_tsne_results[indices, 1], label=cl)

    plt.legend()
    plt.savefig('./experiments/'+model.dataset+'-content-tsne.png')
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-dd', '--data_dir', type=str, default='data')

    args = parser.parse_args()
    main(args)
