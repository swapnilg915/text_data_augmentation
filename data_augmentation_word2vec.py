""" 
date: 23rd dec 2020
aim : text augmentation
description : To augment (duplicate/generate) more text for the minor class in text classification task. It is acheived using python nlp library nlpaug. INthis, we can load word2ec or bert model to generate new sentences. 
Tools/libraries used: nlpaug
reference: https://github.com/makcedward/nlpaug/blob/master/example/textual_language_augmenter.ipynb
"""


import os
import time
import nlpaug.augmenter.word as naw


class W2VAugmenter(object):

	def __init__(self):
		# self.w2v_augmenter = naw.WordEmbsAug(model_type='fasttext', model_path=os.path.join("/home/swapnil/data_folder/Projects/Embeddings/word2vec/fasttext_300_nb.bin"))
		self.w2v_augmenter = naw.WordEmbsAug(model_type='word2vec', model_path=os.path.join("/home/swapnil/data_folder/Projects/Embeddings/word2vec/GoogleNews-vectors-negative300.bin"), action="substitute")
		print("\n w2v model loaded ...")


	def augment_sent(self, text):
		return self.w2v_augmenter.augment(text)


	def main(self, sentences):
		st = time.time()
		print("\n number of sentences : ", len(sentences))
		augmented_sentences = [self.augment_sent(sent) for sent in sentences]
		print("\n number of augmented sentences : ", len(augmented_sentences))
		print("\n total time for augmentation: ", time.time() - st)
		return augmented_sentences


if __name__ == "__main__":
	obj = W2VAugmenter()
	texts = ['The quick brown fox jumps over the lazy dog']

	obj.main(texts)
