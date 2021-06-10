import os
import json
import torch
import argparse

from model_rep import SentenceVae

from utils import to_var, idx2word, interpolate, load_model_params_from_checkpoint


def main(args):

    # load checkpoint
    if not os.path.exists(args.load_checkpoint):    
        raise FileNotFoundError(args.load_checkpoint)

    saved_dir_name = args.load_checkpoint.split('/')[2]
    params_path = './saved_vae_models/'+saved_dir_name+'/model_params.json'
    
    if not os.path.exists(params_path):
        raise FileNotFoundError(params_path)

    # load params
    params = load_model_params_from_checkpoint(params_path)

    # create and load model
    model = SentenceVae(**params)
    print(model)
    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    # load the vocab of the chosen dataset
    
    if(model.dataset == 'yelp'):
        print("Yelp dataset used!")
        
        with open(args.data_dir+'/yelp/yelp.vocab.json', 'r') as file:
            vocab = json.load(file)
    
    w2i, i2w = vocab['w2i'], vocab['i2w']

    samples, z = model.inference(n=args.num_samples)
    print('----------SAMPLES----------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    z1 = torch.randn([params['latent_size']]).numpy()
    z2 = torch.randn([params['latent_size']]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    samples, _ = model.inference(z=z)

    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-p', '--load_params', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)
    parser.add_argument('-dd', '--data_dir', type=str, default='data')

    args = parser.parse_args()
    main(args)
