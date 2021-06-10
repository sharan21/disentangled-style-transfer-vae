import os
import io
import json
import random
import torch
import pickle
import time
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from multiprocessing import cpu_count
from utils import to_var, idx2word, expierment_name
from torch.utils.data import DataLoader
from nltk.tokenize import TweetTokenizer
import collections
import logging
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from gensim.models import KeyedVectors
from collections import OrderedDict, defaultdict
from utils import OrderedCounter
from tqdm import tqdm
import argparse


class Yelp(Dataset):

    def __init__(self, split, create_data=False, have_vocab=False, **kwargs):

        super().__init__()
        self.data_dir = "./data/yelp/"
        self.save_model_path = "./saved_vae_models"
        self.split = split

        if(split == "train"):
            self.num_lines_0 = 176787
            self.num_lines_1 = 267314
        else:
            self.num_lines_0 = 50278
            self.num_lines_1 = 76392

        self.filter_sentiment_words = True
        self.filter_stop_words = True
        self.embedding_size = 300
        self.max_sequence_length = 15
        self.min_occ = kwargs.get('min_occ', 2)    

        self.have_vocab = have_vocab
        self.raw_data_path = "./data/yelp/sentiment." + split + '.'
        self.preprocessed_data_file = 'yelp.'+split+'.json'
        self.vocab_file = 'yelp.vocab.json'
        self.path_to_w2v_embds = './data/yelp/yelp_w2v_embeddings'
        self.path_to_w2v_weights = './data/yelp/yelp_w2v_weights'

        if create_data:
            print("Creating new %s ptb data." % split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.preprocessed_data_file)):
            print("%s preprocessed file not found at %s. Creating new." % (
                split.upper(), os.path.join(self.data_dir, self.preprocessed_data_file)))
            self._create_data()

        else:
            print(" found preprocessed files, no need tooo create data!")
            self._load_data()

        # load bow vocab
        with open("./data/yelp/bow.json") as f:
            self.bow_filtered_vocab_indices = json.load(f)

        self.bow_hidden_dim = len(self.bow_filtered_vocab_indices)

        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'bow': self._get_bow_representations(self.data[idx]['input']),
            # 'label': np.asarray(self.data[idx]['label']),
            'label': np.asarray([1-self.data[idx]['label'], self.data[idx]['label']]), # we need to make it 2 dim to match predicted label dim.
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):

        print("loading preprocessed json data...")

        with open(os.path.join(self.data_dir, self.preprocessed_data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']


    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if not self.have_vocab and self.split == 'train':
            print("creating vocab for train!")
            self._create_vocab()
            print("finished creating vocab!")
            print("creating bow vocab for train!")
            self.create_bow_vocab(self.w2i)
            print("finished creating bow vocab!")           
            print("creating w2v embs matrix")
            self.create_w2v_weight_matrix()
            print("finished creating w2v embs matrix!")
        else:
            self._load_vocab()
            print("loaded vocab from mem!")

        tokenizer = TweetTokenizer(preserve_case=False)
        data = defaultdict(dict)

        labels = ['0', '1']

        for l in labels: 
            print("import data with label {}".format(l))
            file = open(self.raw_data_path + l, 'r')

            num_lines = self.num_lines_0 if l=='0' else self.num_lines_1

            for i, line in enumerate(tqdm(file, total=num_lines)):

                if(i == num_lines):
                    break

                words = tokenizer.tokenize(line)

                # filter out the words greater than this limit
                if(len(words) > self.max_sequence_length):
                    continue

                input = ['<sos>'] + words
                input = input[:self.max_sequence_length]

                target = words[:self.max_sequence_length-1]
                target = target + ['<eos>']

                assert len(input) == len(target), "%i, %i" % (len(input), len(target))
                length = len(input)

                input.extend(['<pad>'] * (self.max_sequence_length-length))
                target.extend(['<pad>'] * (self.max_sequence_length-length))

                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['label'] = int(l)
                data[id]['target'] = target
                data[id]['length'] = length
            
            file.close()

        # shuffle the combined data
        print("Shuffling the combined data!")
        data = self.shuffle(data)

        with io.open(os.path.join(self.data_dir, self.preprocessed_data_file), 'wb') as preprocessed_data_file:
            data = json.dumps(data, ensure_ascii=False)
            preprocessed_data_file.write(data.encode('utf8', 'replace'))


        self._load_data(vocab=False)
    
    def shuffle(self, data):
        
        keys = [i for i in range(len(data))]
        random.shuffle(keys)
        data_shuffled = defaultdict(dict)

        i = 0
        for k in keys:
            if(data[k] is None):
                print("error in shuffle")
                exit()
            data_shuffled[i] = data[k]
            i = i+1

        return data_shuffled

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        self.w2c = OrderedCounter()
        self.w2i = dict()
        self.i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

        for st in special_tokens:
            self.i2w[len(self.w2i)] = st
            self.w2i[st] = len(self.w2i)


        labels = ['0', '1']
        for l in labels:
            print("updating vocab with sentences of label {}".format(l))
            file = open(self.raw_data_path + l, 'r')
            num_lines = self.num_lines_0 if l=='0' else self.num_lines_1

            for i, line in enumerate(tqdm(file, total=num_lines)):
                
                if(i == num_lines):
                    break

                words = tokenizer.tokenize(line)

                if(len(words) > self.max_sequence_length):
                    continue

                self.w2c.update(words)

            file.close()


        print("done creating w2c")
        for w, c in tqdm(self.w2c.items()):
            if c > self.min_occ and w not in special_tokens:
                self.i2w[len(self.w2i)] = w
                self.w2i[w] = len(self.w2i)

        print("done creating w2i")

        assert len(self.w2i) == len(self.i2w)

        print("Vocablurary of %i keys created." % len(self.w2i))

        
        vocab = dict(w2i=self.w2i, i2w=self.i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()
        self.v_size = len(self.w2i)

    def create_w2v_weight_matrix(self):

        self.emb_matrix = np.zeros((self.v_size, self.embedding_size))
        # load the pretrained word embeddings
        w2v_model = KeyedVectors.load_word2vec_format(self.path_to_w2v_embds)

        found = 0
        not_found = 0

        for index in range(self.v_size):
            word = self.i2w[str(index)]
            
            if w2v_model.has_index_for(word):
                self.emb_matrix[index] = w2v_model.get_vector(word)
                found += 1
            else:
                self.emb_matrix[index] = np.random.randn(self.embedding_size)
                # print("word: {} was not found ".format(word))
                not_found += 1

        np.save(self.path_to_w2v_weights, self.emb_matrix)
        print("Done creating w2v embedding matrix. {} found and {} unfound".format(found, not_found))


   
    def _get_bow_representations(self, text_sequence):
        """
        Returns BOW representation of every sequence of the batch
        """
        # self.bow_hidden_dim = len(self.bow_filtered_vocab_indices)
        sequence_bow_representation = np.zeros(shape=self.bow_hidden_dim, dtype=np.float32)
     
        # Iterate over each word in the sequence
        for index in text_sequence:

            if str(index) in self.bow_filtered_vocab_indices:
                bow_index = self.bow_filtered_vocab_indices[str(index)]
                sequence_bow_representation[bow_index] += 1
        
        # removing normalisation because the loss becomes too low with it, anyway it wont change correctness
        sequence_bow_representation /= np.max([np.sum(sequence_bow_representation), 1])

        return np.asarray(sequence_bow_representation)

    def create_bow_vocab(self, word_index):
        """
        Creates a dict of vocab indeces of non-stopwords and non-sentiment words
        """
        blacklisted_words = set()
        bow_filtered_vocab_indices = dict()
        # The '|' operator on sets in python acts as a union operator
        # blacklisted_words |= set(self.predefined_word_index.values())
        if self.filter_sentiment_words:
            blacklisted_words |= self._get_sentiment_words()
        if self.filter_stop_words:
            blacklisted_words |= self._get_stopwords()

        allowed_vocab = word_index.keys() - blacklisted_words
        i = 0

        for word in allowed_vocab:
            vocab_index = word_index[word]
            bow_filtered_vocab_indices[vocab_index] = i
            i += 1

        self.bow_hidden_dim = len(allowed_vocab)
        print("Created word index blacklist for BoW")
        print("BoW size: {}".format(self.bow_hidden_dim))
        
        # saving bow vocab
        with open('./data/yelp/bow.json', 'w') as json_file:
            json.dump(bow_filtered_vocab_indices, json_file)
        
        print("Saved bow.json at {}".format('./data/yelp/bow.json'))

    def _get_sentiment_words(self):
        """
        Returns all the sentiment words (positive and negative)
        which are excluded from the main vocab to form the BoW vocab
        """
        with open(file='./data/lexicon/positive-words.txt', mode='r', encoding='ISO-8859-1') as pos_sentiment_words_file,\
            open(file='./data/lexicon/negative-words.txt', mode='r', encoding='ISO-8859-1') as neg_sentiment_words_file:
            pos_words = pos_sentiment_words_file.readlines()
            neg_words = neg_sentiment_words_file.readlines()
            words = pos_words + neg_words
        words = set(word.strip() for word in words)

        return words

    def _get_stopwords(self):
        """
        Returns all the stopwords which excluded from the
        main vocab to form the BoW vocab
        """
        nltk_stopwords = set(stopwords.words('english'))
        sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS

        all_stopwords = set()
        # The '|' operator on sets in python acts as a union operator
        all_stopwords |= spacy_stopwords
        all_stopwords |= nltk_stopwords
        all_stopwords |= sklearn_stopwords

        return all_stopwords

if __name__ == "__main__":
    # prepare dataset
    splits = ['train', 'test']
    # splits = ['train']

    # create dataset object
    datasets = OrderedDict()

    # create test and train split in data, also preprocess
    for split in splits:
        # print("creating dataset for: {}".format(split))
        datasets[split] = Yelp(
            split=split,
            create_data=False,
            min_occ=2
        )