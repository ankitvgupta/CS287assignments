#!/usr/bin/env python

"""Classes for binary features of sentences
"""

class SentenceFeature(object):

	def __init__(self, dwin):
		self.dwin = dwin

	def initialize(self):
		pass

	def convert(self, padded_sentence, word_idx):
		raise UnimplementedError

	def isSparse(self):
		raise UnimplementedError

	def validate_sentence(self, padded_sentence, word_idx):
		l = len(padded_sentence)
		assert word_idx >= self.dwin/2 and word_idx < l-self.dwin/2
		for i in range(0, self.dwin/2):
			assert padded_sentence[i] == "PADDING"
			assert padded_sentence[l-i-1] == "PADDING"


class UnigramFeature(SentenceFeature):
	def __init__(self, dwin, vocab):
		self.dwin = dwin
		self.vocab = vocab
		self.word_to_idx = {}

	def initialize(self):
		for i, word in enumerate(self.vocab):
			self.word_to_idx[word] = i+1

		self.numWords = len(self.vocab)

	def convert(self, padded_sentence, word_idx):
		self.validate_sentence(padded_sentence, word_idx)

		feat = []
		start = word_idx - self.dwin/2
		end = word_idx + self.dwin/2
		for i in range(start, end+1):
			word = padded_sentence[i].lower()
			try:
				feat.append(self.word_to_idx[word]+i*self.numWords)
			except KeyError:
				feat.append(self.word_to_idx["RARE"]+i*self.numWords)

		return feat

	def isSparse(self):
		return True


class CapitalizationFeature(SentenceFeature):

	def convert(self, padded_sentence, word_idx):
		self.validate_sentence(padded_sentence, word_idx)

		feat = []
		start = word_idx - self.dwin/2
		end = word_idx + self.dwin/2
		for i in range(start, end+1):
			word = padded_sentence[i]
			upper = int(word[0].isupper())
			feat.append(upper)

		return feat

	def isSparse(self):
		return False
		