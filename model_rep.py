import torch
import torch.nn as nn

import numpy as np
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
from torch.autograd import Variable


class SentenceVae(nn.Module):
	def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, latent_size,
				sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, content_bow_dim, path_to_w2v_weights=None, 
				num_layers=1, dataset='yelp', bidirectional=False):

		super().__init__()

		#this implementation does not support bidirectional or multi layer models

		self.path_to_w2v_weights = path_to_w2v_weights
		self.style_space_size = 8
		self.content_space_size = 128
		self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
		self.content_bow_dim = content_bow_dim
		self.max_sequence_length = max_sequence_length

		self.sos_idx = sos_idx
		self.eos_idx = eos_idx
		self.pad_idx = pad_idx
		self.unk_idx = unk_idx

		self.latent_size = self.style_space_size + self.content_space_size
		self.dataset = dataset
		self.rnn_type = rnn_type
		self.bidirectional = bidirectional
		self.num_layers = num_layers
		self.hidden_size = hidden_size 
		
		if(self.dataset == 'yelp'):
			self.output_size = 2
		
		if(self.dataset == 'snli'):
			self.output_size = 2
		
		if(self.dataset == 'multitask'):
			self.output_size = 4

		# load pretrained w2v weights if available
		if(path_to_w2v_weights is not None):
			print("w2v embedding weights loaded!")
			weights = torch.FloatTensor(np.load(path_to_w2v_weights))
			self.embedding = nn.Embedding.from_pretrained(weights)
			print("w2v embedding weights loaded!")
		else:
			self.embedding = nn.Embedding(vocab_size, embedding_size)

		if rnn_type == 'rnn':
			rnn = nn.RNN
		elif rnn_type == 'gru':
			rnn = nn.GRU
		else:
			raise ValueError()
			
		self.word_dropout = word_dropout
		self.dropout = nn.Dropout(self.word_dropout)

		self.encoder = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
		self.decoder = rnn(embedding_size+self.latent_size, hidden_size, num_layers=num_layers, batch_first=True)

		######## hidden to style space ########
		self.hidden2stylemean = nn.Linear(hidden_size, self.style_space_size)
		self.hidden2stylelogv = nn.Linear(hidden_size, self.style_space_size)

		######### hidden to content space#######
		self.hidden2contentmean = nn.Linear(hidden_size, self.content_space_size)
		self.hidden2contentlogv = nn.Linear(hidden_size, self.content_space_size)

		########## multitasl classifiers ############
		self.content_classifier = nn.Linear(self.content_space_size, self.content_bow_dim)
		self.style_classifier = nn.Linear(self.style_space_size, self.output_size) 
		
		############ adversaries/disc ###########
		self.style_adv_classifier = nn.Linear(self.style_space_size, self.content_bow_dim)
		self.content_adv_classifier = nn.Linear(self.content_space_size, self.output_size)

		######### latent to initial hs for decoder ########
		self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size)

		###### final hidden to output vocab #########
		self.outputs2vocab = nn.Linear(self.hidden_size, vocab_size, bias = False)

		###### extra misc parameters ########
		
		self.label_smoothing = 0.1
		
		

	def forward(self, input_sequence, length, labels, content_bow):

		batch_size = input_sequence.size(0) #get batch size
		input_embedding = self.embedding(input_sequence) # convert to embeddings
		
		################### encoder ##################
		_, hidden = self.encoder(self.dropout(input_embedding)) # hidden -> (B, H)
	
		###### if the RNN has multiple layers, take mean 
		if self.bidirectional or self.num_layers > 1:
			hidden = torch.mean(hidden, axis=0) #take the mean of both bi layers
		else:
			hidden = hidden.squeeze()

		############## REPARAMETERIZATION of style and content###############

		############style component############

		style_mean = self.hidden2stylemean(hidden) #calc latent mean 
		style_logv = self.hidden2stylelogv(hidden) #calc latent variance
		style_std = torch.exp(0.5 * style_logv) #find sd

		style_random = torch.randn([batch_size, self.style_space_size])

		style_z = to_var(style_random) #get a random vector
		style_z = style_z * torch.exp(style_logv) + style_mean #copmpute datapoint

		############content component###############

		content_mean = self.hidden2contentmean(hidden) #calc latent mean 
		content_logv = self.hidden2contentlogv(hidden) #calc latent variance
		content_std = torch.exp(0.5 * content_logv) #find sd

		content_random = torch.randn([batch_size, self.content_space_size])

		content_z = to_var(content_random) #get a random vector
		content_z = content_z * torch.exp(content_logv) + content_mean #compute datapoint

		#############concat style and concat
		
		final_mean = torch.cat((style_mean, content_mean), dim=1)
		final_logv = torch.cat((style_logv, content_logv), dim=1)
		final_z = torch.cat((style_z, content_z), dim=1)

		##########3#####style and content multitask classifiers###################
		
		style_preds = nn.Softmax(dim=1)(self.style_classifier(self.dropout(style_z)))
		content_preds = nn.Softmax(dim=1)(self.content_classifier(self.dropout(content_z)))

		################style and content adversarial discriminators###################
		# we detach() so that the gradients dont forward prop back to encoder
		# check if we also need to take softmax and check loss function works the best

		style_adv_preds = nn.Softmax(dim=1)(self.style_adv_classifier(self.dropout(style_z.detach())))
		content_adv_preds = nn.Softmax(dim=1)(self.content_adv_classifier(self.dropout(content_z.detach())))

		######################## DECODER
		hidden = self.latent2hidden(final_z)

		############# format decoder input for dropout
		if self.word_dropout > 0:
			
			# randomly replace decoder input with <unk>
			prob = torch.rand(input_sequence.size())
			
			if torch.cuda.is_available():
				prob=prob.cuda()

			prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
			decoder_input_sequence = input_sequence.clone()
			decoder_input_sequence[prob < self.word_dropout] = self.unk_idx
			input_embedding = self.embedding(decoder_input_sequence)

		############### decoder forward pass

		# concat latent vector to input embds
		b, s, _ = input_embedding.size()
		final_z_copy = final_z.detach().unsqueeze(axis=1)
		final_z_copy = final_z_copy.expand(b, s, self.latent_size)
		input_to_decoder = torch.cat((input_embedding, final_z_copy), axis = 2)

		hidden = hidden.unsqueeze(axis=0)
		outputs, _ = self.decoder(input_to_decoder, hidden)
		outputs = outputs.contiguous()
		b, s, _ = outputs.size()

		################ project outputs to vocab
		logp = self.outputs2vocab(outputs.view(-1, outputs.size(2)))
		logp = nn.functional.log_softmax(logp, dim=-1)
		logp = logp.view(b, s, self.embedding.num_embeddings)

		return logp, style_mean, style_logv, content_mean, content_logv, style_preds, content_preds, style_adv_preds, content_adv_preds


	def inference(self, n=4, z=None):
		'''Generate words from Latent vectors using top-k/greedy sampling without teacher forcing. '''

		if z is None:
			batch_size = n
			z = to_var(torch.randn([batch_size, self.latent_size]))
		else:
			batch_size = z.size(0)

		hidden = self.latent2hidden(z)

		if self.num_layers > 1:
			# unflatten hidden state
			hidden = hidden.view(self.num_layers, batch_size, self.hidden_size)

		hidden = hidden.unsqueeze(0)

		# required for dynamic stopping of sentence generation
		sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
		# all idx of batch which are still generating
		sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
		sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
		# idx of still generating sequences with respect to current loop
		running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

		generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

		t = 0
		while t < self.max_sequence_length and len(running_seqs) > 0:

			if t == 0:
				input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

			input_sequence = input_sequence.unsqueeze(1)
			input_embedding = self.embedding(input_sequence)
			
			b, s, _ = input_embedding.size()
			final_z_copy = z.detach().unsqueeze(axis=1)
			final_z_copy = final_z_copy.expand(batch_size, s, self.latent_size)
			input_to_decoder = torch.cat((input_embedding, final_z_copy[0:b, :, :]), axis = 2)
			output, hidden = self.decoder(input_to_decoder, hidden)
			
			logits = self.outputs2vocab(output)
			input_sequence = self._sample(logits)

			# save next input
			generations = self._save_sample(generations, input_sequence, sequence_running, t)

			# update gloabl running sequence
			sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
			sequence_running = sequence_idx.masked_select(sequence_mask)

			# update local running sequences
			running_mask = (input_sequence != self.eos_idx).data
			running_seqs = running_seqs.masked_select(running_mask)

			# prune input and hidden state according to local update
			if len(running_seqs) > 0:
				input_sequence = input_sequence[running_seqs]
				hidden = hidden[:, running_seqs]
				running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

			t += 1

		return generations, z

	def final_z_to_words(self, input_sequence, final_z):
		'''Generate words using teacher forcing from the latent vector. Used in inference.'''

		input_sequence = torch.tensor(input_sequence).cuda().unsqueeze(0)
		input_embedding = self.embedding(input_sequence) # convert to embeddings
		
		######################## DECODER
		hidden = self.latent2hidden(final_z)

		if self.bidirectional or self.num_layers > 1:
			# unflatten hidden state
			pass
			# hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
		else:
			hidden = hidden.unsqueeze(0)

		############### decoder forward pass
		outputs, _ = self.decoder(input_embedding, hidden)

		outputs = outputs.contiguous()
		b,s,_ = outputs.size()

		################ project outputs to vocab
		logp = nn.functional.log_softmax(self.outputs2vocab(outputs.view(-1, outputs.size(2))), dim=-1)
		logp = logp.view(b, s, self.embedding.num_embeddings)
		words = torch.argmax(logp, axis = -1)

		return words

	def get_style_content_space(self, input_sequence):

		batch_size = input_sequence.size(0) #get batch size
		input_embedding = self.embedding(input_sequence) # convert to embeddings
		
		################### encoder ##################
		_, hidden = self.encoder(input_embedding) # hidden -> (B, H)
	
		###### if the RNN has multiple layers, flatten all the hiddens states 
		if self.bidirectional or self.num_layers > 1:
			hidden = torch.mean(hidden, axis = 1)
		else:
			hidden = hidden.squeeze()

		############## REPARAMETERIZATION of style and content###############

		############style component############

		style_mean = self.hidden2stylemean(hidden) #calc latent mean 
		style_logv = self.hidden2stylelogv(hidden) #calc latent variance
		style_std = torch.exp(0.5 * style_logv) #find sd

		style_z = to_var(torch.randn([batch_size, self.style_space_size])) #get a random vector
		style_z = style_z * torch.exp(style_logv) + style_mean #copmpute datapoint

		############content component###############

		content_mean = self.hidden2contentmean(hidden) #calc latent mean 
		content_logv = self.hidden2contentlogv(hidden) #calc latent variance
		content_std = torch.exp(0.5 * content_logv) #find sd

		content_z = to_var(torch.randn([batch_size, self.content_space_size])) #get a random vector
		content_z = content_z * torch.exp(content_logv) + content_mean #compute datapoint

		return style_z, content_z

	def encode_to_lspace(self, input_sequence, zero_noise=False):

		'''Takes a batch of sentence tokens (B,L,E) and converts them to latent vectors (B, LS). Used in inference'''

		batch_size = input_sequence.size(0) #get batch size
		input_embedding = self.embedding(input_sequence) # convert to embeddings
		
		################### encoder ##################
		_, hidden = self.encoder(input_embedding) # hidden -> (B, H)
		####### if the RNN has multiple layers, flatten all the hiddens states 
		if self.bidirectional or self.num_layers > 1:
			hidden = torch.mean(hidden, axis=0)
		else:
			hidden = hidden.squeeze()

		############style component

		style_mean = self.hidden2stylemean(hidden) #calc latent mean 
		style_logv = self.hidden2stylelogv(hidden) #calc latent variance
		style_std = torch.exp(0.5 * style_logv) #find sd

		if(zero_noise):
			style_z = style_mean
		else:
			style_z = to_var(torch.randn([batch_size, self.style_space_size])) #get a random vector
			style_z = style_z * torch.exp(style_logv) + style_mean #copmpute datapoint

		############content component

		content_mean = self.hidden2contentmean(hidden) #calc latent mean 
		content_logv = self.hidden2contentlogv(hidden) #calc latent variance
		content_std = torch.exp(0.5 * content_logv) #find sd

		if(zero_noise):
			content_z = content_mean
		else:
			content_z = to_var(torch.randn([batch_size, self.content_space_size])) #get a random vector
			content_z = content_z * torch.exp(content_logv) + content_mean #compute datapoint

		#############concat style and concat
		
		final_mean = torch.cat((style_mean, content_mean), dim=1)
		final_logv = torch.cat((style_logv, content_logv), dim=1)
		final_z = torch.cat((style_z, content_z), dim=1)

		return style_z, content_z

	def _sample(self, dist, mode='greedy'):

		if mode == 'greedy':
			_, sample = torch.topk(dist, 1, dim=-1)
		sample = sample.reshape(-1)

		return sample

	def _save_sample(self, save_to, sample, running_seqs, t):
		# select only still running
		running_latest = save_to[running_seqs]
		# update token at position t
		running_latest[:,t] = sample.data
		# save back
		save_to[running_seqs] = running_latest

		return save_to
