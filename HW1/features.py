#!/usr/bin/env python

"""Classes for binary features of sentences
"""
from textblob import Blobber, TextBlob, Word
from textblob.taggers import PatternTagger

class SentenceFeature(object):

	def __init__(self, index_offset=2):
		self.index_offset = index_offset

	def initialize(self, sentences):
		pass

	def sentenceToFeatures(self, sentence):
		raise UnimplementedError

	def totalFeatureCount(self):
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

		return list(set(feat))

	def totalFeatureCount(self):
		return len(self.word_to_idx)

	def maxFeatureLength(self):
		return self.max_sent_len


# four features: 
# first is high polarity, second is low polarity
# third is high subjectivity, fourth is low subjectivity
class SentimentFeature(SentenceFeature):

	def __init__(self, neutral_width=0.2, index_offset=2):
		self.neutral_width = neutral_width
		self.index_offset = index_offset

	def sentenceToFeatures(self, sentence):
		sentence_str = ' '.join(sentence)
		blob = TextBlob(sentence_str)
		feats = []

		# polarity comes between -1 and 1, so normalize it
		polarity = (blob.sentiment.polarity+1.0)/2
		if polarity > (0.5 + self.neutral_width):
			feats.append(self.index_offset)
		elif polarity < (0.5 - self.neutral_width):
			feats.append(self.index_offset+1)

		subjectivity = blob.sentiment.subjectivity
		if subjectivity > (0.5 + self.neutral_width):
			feats.append(self.index_offset+2)
		elif subjectivity < (0.5 - self.neutral_width):
			feats.append(self.index_offset+3)

		return feats

	def totalFeatureCount(self):
		return 4

	def maxFeatureLength(self):
		return 2

# checks for presence for parts of speech
class POSFeature(SentenceFeature):

	def initialize(self, sentences):
		self.tagger = Blobber(pos_tagger=PatternTagger())
		parts_of_speech = ['DT', 'NN', 'VBZ', 'TO', 'VB', 'CD', 'POS', 'JJ', 'CC', 'IN', 'PRP', 'VBG', 'RB', 'JJR', 'NNS', 'MD']
		self.pos_to_idx = {}
		for i,pos in enumerate(parts_of_speech):
			self.pos_to_idx[pos] = i+self.index_offset

	def sentenceToFeatures(self, sentence):
		feats = []
		sentence_str = ' '.join(sentence)
		blob = self.tagger(sentence_str)
		for _, pos in blob.tags:
			try:
				feats.append(self.pos_to_idx[pos])
			except KeyError:
				continue
		return list(set(feats))

	def totalFeatureCount(self):
		return len(self.pos_to_idx)

	def maxFeatureLength(self):
		return len(self.pos_to_idx)


class SynFeature(SentenceFeature):

	def initialize(self, sentences):
		self.max_feat_len = 0
		self.word_to_idx = {}
		idx = self.index_offset
		for sentence in sentences:
			syn_count = 0
			for word in sentence:
				werd = Word(word)
				syns = [w.lemma_names for w in werd.get_synsets()]
				for syn in syns:
					syn_count += 1
					if syn not in self.word_to_idx:
						self.word_to_idx[syn] = idx
						idx += 1
			self.max_feat_len = max(self.max_feat_len, syn_count)

	def sentenceToFeatures(self, sentence):
		feat = []

		for word in sentence:
			werd = Word(word)
			syns = [w.lemma_names for w in werd.get_synsets()]
			for syn in syns:
				try:
					feat.append(self.word_to_idx[syn])
				except KeyError:
					continue
		return list(set(feat))

	def totalFeatureCount(self):
		return len(self.word_to_idx)

	def maxFeatureLength(self):
		return self.max_feat_len
		