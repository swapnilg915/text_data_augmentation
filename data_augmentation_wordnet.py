import os
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

text = 'The quick brown fox jumps over the lazy dog'

### wordnet
aug = naw.SynonymAug(aug_src='wordnet', lang='eng')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)