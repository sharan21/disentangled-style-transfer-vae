import torch
import torch.nn as nn

import numpy as np
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
from torch.autograd import Variable

# this class implements a simple classifier which is used as one of the quantitative metrics to measure style transfer
class BinaryClassifier(nn.Module):
	def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size,
				sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, dataset='snli', bidirectional=False, 
				attention=False):

		super().__init__()

		#this implementation does not support bidirectional or multi layer models
		
		self.attention = attention
		self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
		
		self.max_sequence_length = max_sequence_length

		self.sos_idx = sos_idx
		self.eos_idx = eos_idx
		self.pad_idx = pad_idx
		self.unk_idx = unk_idx

		self.dataset = dataset
		self.rnn_type = rnn_type
		self.bidirectional = bidirectional # need to be implemented
		self.num_layers = num_layers
		self.hidden_size = hidden_size 
		
		if(self.dataset == 'yelp'):
			self.output_size = 2
		
		if(self.dataset == 'snli'):
			self.output_size = 2
		
		if(self.dataset == 'multitask'):
			self.output_size = 4

		self.embedding = nn.Embedding(vocab_size, embedding_size)
	
		if rnn_type == 'rnn':
			rnn = nn.RNN
		elif rnn_type == 'gru':
			rnn = nn.GRU
		else:
			raise ValueError()

		self.encoder = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
		self.final_layer = nn.Linear(self.hidden_size, self.output_size)
		
		
  
	def self_attention(self, lstm_output, final_state):

		# lstm_output : L*B*H
		# final_state : L*B*1
		
		# reshaping to satisfy torch.bmm
		lstm_output_2 = lstm_output.permute(1,0,2) #B*L*H
		final_state_2 = final_state.permute(1,2,0) #B*1*L
		
		#get attention scores
		attn_weights = torch.bmm(lstm_output_2, final_state_2) #B*L, dot product attention
		soft_attn_weights = F.softmax(attn_weights, 1) #B*L
		
		# weighted sum to get final attention vector
		lstm_output_2 = lstm_output.permute(1,0,2) #B*L*H, needed for mat mul in next step
		new_hidden_state = lstm_output_2 * soft_attn_weights #B*L*H * B*L = #B*L*H
		new_hidden_state = torch.sum(new_hidden_state, axis=1) #B*H
		
		return new_hidden_state


	def forward(self, input_sequence):

		batch_size = input_sequence.size(0) #get batch size
		input_embedding = self.embedding(input_sequence) # convert to embeddings
		
		################### encoder ##################
		_, hidden = self.encoder(input_embedding) # hidden -> (B, H)
	
		###### if the RNN has multiple layers, flatten all the hiddens states 
		if self.bidirectional or self.num_layers > 1:
			hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor) # flatten hidden state
		else:
			hidden = hidden.squeeze()

		####### self attention
		if(self.attention):	
			hidden = self.self_attention(output, hidden)
		else:
			pass
			# hidden = hidden[-1] # take the last hidden state of lstm

		####### if the RNN has multiple layers, flatten all the hiddens states 
		if self.bidirectional or self.num_layers > 1:
			hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor) # flatten hidden state
		else:
			hidden = hidden.squeeze()

		# print(hidden.shape)
		preds = nn.Softmax(dim=1)(self.final_layer(hidden))

		return preds

		

	

		
	
