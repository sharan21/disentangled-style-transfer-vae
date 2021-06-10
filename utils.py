import torch
import numpy as np
from torch.autograd import Variable
from collections import defaultdict, Counter, OrderedDict
import os
import kenlm
import argparse
import json
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def get_lm_score(model_path, sentences):
    model = kenlm.LanguageModel(model_path)
    return model.score(sentence)


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def load_model_params_from_checkpoint(path_to_params):
    with open(path_to_params) as json_file:
        params = json.load(json_file)
    
    return params


def idx2word(idx, i2w, pad_idx):
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "
        sent_str[i] = sent_str[i].strip()
    return sent_str


def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps+2)

    return interpolation.T


def expierment_name(args, ts):
    exp_name = str()
    exp_name += "BS=%i_" % args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_" % args.embedding_size
    exp_name += "%s_" % args.rnn_type.upper()
    exp_name += "HS=%i_" % args.hidden_size
    exp_name += "L=%i_" % args.num_layers
    exp_name += "BI=%i_" % args.bidirectional
    exp_name += "LS=%i_" % args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_" % args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_" % args.x0
    exp_name += "TS=%s" % ts

    return exp_name

def convert_sentences_to_latent(args):

    # load vocab and mapping dictionaries
    with open('./data/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    # init the model
    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    # samples, z = model.inference(n=args.num_samples)
    # print('----------SAMPLES----------')
    # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    # z1 = torch.randn([args.latent_size]).numpy()
    # z2 = torch.randn([args.latent_size]).numpy()
    # z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    # samples, _ = model.inference(z=z)
    # print('-------INTERPOLATION-------')
    # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')




def train_word2vec_model(text_file_path, model_file_path, embedding_size):
    # define training data
    # train model
    print("Loading input file and training mode ...")
    model = Word2Vec(sentences=LineSentence(text_file_path), min_count=1, vector_size=embedding_size)
    # summarize the loaded model
    print("Model Details: {}".format(model))
    # save model
    model.wv.save_word2vec_format(model_file_path, binary=False)
    print("Model saved")


    
    

