import gensim
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict








class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epochs = []
        self.epoch = 1
        self.losses = []
        self.deltas = []
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            delta = loss
        else:
            delta = loss- self.loss_previous_step
        self.loss_previous_step=loss
        self.losses.append(loss)
        self.epochs.append(self.epoch)
        self.epoch += 1
        self.deltas.append(delta)





class TokenSimilarities():

	def __init__(self, path):

		# The aspects of the model that are actually used in the context of this class.
		# This prepares as a nested dictionary that holds pairwise word similarities in a sparse way, so they can be looked up rather than found from comparing embeddings.
		self.model = gensim.models.Word2Vec.load(path)
		self.vocabulary = self.model.wv.vocab
		self.pairwise_token_similarities = defaultdict(dict)

		# The value of n here determines how many similarities to a given token are actually calculated. 
		# The rest of the [vocabulary size]-n similarity values for a given token are considered to be zero.
		# The larger the value of n, the more information to the model is incorporated, but many of these word similarities are irrelevant,
		# and increasing n makes the preprocessing take much longer as n approaches the size of the full vocabulary.
		n=50
		maximum_possible_similarity = 1.00
		assert n<len(self.vocabulary)
		for token in self.vocabulary:
			self.pairwise_token_similarities[token][token] = maximum_possible_similarity
			n_most_similar_tokens = self.model.most_similar(token, topn=n)
			for similar_token,similarity in n_most_similar_tokens:
				self.pairwise_token_similarities[token][similar_token] = similarity
				self.pairwise_token_similarities[similar_token][token] = similarity



	def similarity(self, token1, token2):
		return(self.pairwise_token_similarities[token1].get(token2,0.00))



	def get_mean_embedding(self, tokens):
		return(np.array(np.mean([self.model[token] for token in tokens if token in self.vocabulary], axis=0)))

