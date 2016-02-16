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

	def numFeats(self):
		raise UnimplementedError


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
		for idx, i in enumerate(range(start, end+1)):
			word = padded_sentence[i]
			if word is not "PADDING":
				word = word.lower()
			try:
				feat.append(self.word_to_idx[word])
			except KeyError:
				feat.append(self.word_to_idx["RARE"])

		return feat

	def isSparse(self):
		return True

	def numFeats(self):
		return self.numWords


class CapitalizationFeature(SentenceFeature):

	# for each word,
	# first entry is all lowercase
	# second entry is all caps
	# third entry is first letter capital
	# fourth entry is one letter capital
	def convert(self, padded_sentence, word_idx):
		self.validate_sentence(padded_sentence, word_idx)

		feat = []
		start = word_idx - self.dwin/2
		end = word_idx + self.dwin/2
		for i in range(start, end+1):
			word = padded_sentence[i]
			all_lower = int(word.islower())
			all_caps = int(word.isupper())
			first_cap = int(word[0].isupper())
			one_letter_cap = int(len([c for c in word if c.isupper()]) == 1)
			feat.extend([all_lower, all_caps, first_cap, one_letter_cap])

		return feat

	def isSparse(self):
		return False

	def numFeats(self):
		return 4*self.dwin
