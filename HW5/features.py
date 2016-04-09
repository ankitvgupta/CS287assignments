#!/usr/bin/env python

"""Classes for binary features of words for NER
"""
from nltk.stem.wordnet import WordNetLemmatizer
from stemming.porter2 import stem as porterStem
from textblob import Blobber, TextBlob, Word
from textblob.taggers import PatternTagger

class Feature(object):

	def __init__(self, dwin, index_offset):
		self.dwin = dwin
		self.index_offset = index_offset

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
			assert padded_sentence[i] == "<FRONTPADDING>"
			assert padded_sentence[l-i-1] == "<BACKPADDING>"

	def maxFeatIdx(self):
		raise UnimplementedError

	def numFeats(self):
		raise UnimplementedError


class UnigramFeature(Feature):
	def __init__(self, dwin, index_offset, vocab):
		self.dwin = dwin
		self.index_offset = index_offset
		self.vocab = vocab
		self.word_to_idx = {}

	def initialize(self):
		for i, word in enumerate(self.vocab):
			self.word_to_idx[word] = self.index_offset+i+1

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

	def maxFeatIdx(self):
		return max(self.word_to_idx.values())

	def numFeats(self):
		return self.numWords


# THIS DOESN'T SEEM TO BE MUCH DIFFERENT THAN UNIGRAM
class LemmaFeature(Feature):
	def __init__(self, dwin, index_offset, vocab):
		self.dwin = dwin
		self.index_offset = index_offset
		self.vocab = vocab
		self.lemma_to_idx = {}
		self.lmtzr = WordNetLemmatizer()

	def _getLemma(self, word):
		return self.lmtzr.lemmatize(word)

	def initialize(self):
		for i, word in enumerate(self.vocab):
			lemma = self._getLemma(word)
			self.lemma_to_idx[lemma] = self.index_offset+i+1

		self.numLemmas = len(self.lemma_to_idx)
		self.rare_idx = max(self.lemma_to_idx.values()) + 1

	def convert(self, padded_sentence, word_idx):
		self.validate_sentence(padded_sentence, word_idx)

		feat = []
		start = word_idx - self.dwin/2
		end = word_idx + self.dwin/2
		for idx, i in enumerate(range(start, end+1)):
			word = padded_sentence[i]
			if word is not "PADDING":
				word = word.lower()
			lemma = self._getLemma(word)
			try:
				feat.append(self.lemma_to_idx[lemma])
			except KeyError:
				feat.append(self.rare_idx)

		return feat

	def isSparse(self):
		return True

	def maxFeatIdx(self):
		return self.rare_idx

	def numFeats(self):
		return self.numLemmas


class StemFeature(Feature):
	def __init__(self, dwin, index_offset, vocab):
		self.dwin = dwin
		self.index_offset = index_offset
		self.vocab = vocab
		self.stem_to_idx = {}

	def _getStem(self, word):
		return porterStem(word)

	def initialize(self):
		for i, word in enumerate(self.vocab):
			stem = self._getStem(word)
			self.stem_to_idx[stem] = self.index_offset+i+1

		self.numStems = len(self.stem_to_idx)
		self.rare_idx = max(self.stem_to_idx.values()) + 1

	def convert(self, padded_sentence, word_idx):
		self.validate_sentence(padded_sentence, word_idx)

		feat = []
		start = word_idx - self.dwin/2
		end = word_idx + self.dwin/2
		for idx, i in enumerate(range(start, end+1)):
			word = padded_sentence[i]
			if word is not "PADDING":
				word = word.lower()
			stem = self._getStem(word)
			try:
				feat.append(self.stem_to_idx[stem])
			except KeyError:
				feat.append(self.rare_idx)

		return feat

	def isSparse(self):
		return True

	def maxFeatIdx(self):
		return self.rare_idx

	def numFeats(self):
		return self.numStems



class POSFeature(Feature):

	def initialize(self):
		self.tagger = Blobber(pos_tagger=PatternTagger())
		parts_of_speech = ['DT', 'NN', 'VBZ', 'TO', 'VB', 'CD', 'POS', 'JJ', 'CC', 'IN', 'PRP', 'VBG', 'RB', 'JJR', 'NNS', 'MD']
		self.pos_to_idx = {}
		for i,pos in enumerate(parts_of_speech):
			self.pos_to_idx[pos] = i+self.index_offset
		self.rare_idx = max(self.pos_to_idx.values())+1

	def recover_original_sentence(self, padded_sentence):
		start = self.dwin/2
		end = len(padded_sentence)-self.dwin/2
		center = padded_sentence[start:end]
		sent_str = ' '.join(center)
		return sent_str


	def convert(self, padded_sentence, word_idx):
		self.validate_sentence(padded_sentence, word_idx)

		sentence_str = self.recover_original_sentence(padded_sentence)
		blob = self.tagger(sentence_str)

		padded_idx = -1
		pad_word = None
		l = len(padded_sentence)
		parts_of_speech = [None for _ in range(l)]

		for (word, pos) in blob.tags:
			last_found_idx = padded_idx
			found_new_word = True
			while word != pad_word:
				padded_idx += 1
				# word does not exist in original sentence
				if padded_idx >= l:
					padded_idx = last_found_idx
					found_new_word = False
					break

				pad_word = padded_sentence[padded_idx]
			
			if found_new_word:
				parts_of_speech[padded_idx] = pos

		feat = []
		start = word_idx - self.dwin/2
		end = word_idx + self.dwin/2

		for idx, i in enumerate(range(start, end+1)):
			pos = parts_of_speech[i]
			try:
				feat.append(self.pos_to_idx[pos])
			except KeyError:
				feat.append(self.rare_idx)

		return feat

	def isSparse(self):
		return True

	def maxFeatIdx(self):
		return self.rare_idx

	def numFeats(self):
		return len(self.pos_to_idx)



class CapitalizationFeature(Feature):

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
			all_lower = 1+int(word.islower())
			all_caps = 1+int(word.isupper())
			first_cap = 1+int(word[0].isupper())
			one_letter_cap = 1+int(len([c for c in word if c.isupper()]) == 1)
			feat.extend([all_lower, all_caps, first_cap, one_letter_cap])

		return feat

	def isSparse(self):
		return False

	def numFeats(self):
		return 4*self.dwin
