#!/usr/bin/env python

"""Classes for binary features of sentences
"""

class SentenceFeature(object):

	def __init__(self, index_offset=2):
		self.index_offset = index_offset

	def initialize(self, sentences=None):
		raise UnimplementedError

	def sentenceToFeatures(self, sentence):
		raise UnimplementedError

	def maxFeatureLength(self):
		raise UnimplementedError


class NgramFeature(SentenceFeature):

	def __init__(self, N=1, index_offset=2):
		self.N = N
		self.index_offset = index_offset

	def initialize(self, sentences):
		self.max_sent_len = 0
		self.word_to_idx = {}
		idx = self.index_offset
		for sentence in sentences:
			l = len(sentence)-(self.N-1)
			self.max_sent_len = max(self.max_sent_len, l)
			for start_idx in range(0, l):
				gram = ' '.join(sentence[start_idx:start_idx+self.N])
				if gram not in self.word_to_idx:
				    self.word_to_idx[gram] = idx
				    idx += 1

	def sentenceToFeatures(self, sentence):
		feat = []

		l = len(sentence)-(self.N-1)
		for start_idx in range(0, l):
			gram = ' '.join(sentence[start_idx:start_idx+self.N])
			try:
				feat.append(self.word_to_idx[gram])
			except KeyError:
				continue
		return feat

	def maxFeatureLength(self):
		return self.max_sent_len
