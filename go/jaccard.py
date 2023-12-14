# Overlap of `significant heads' between the the finetuned-pretrained pair, the finetuned-shuffled pair and the pretrained-shuffled pair was quantified with the Jaccard similarity coefficient
 
import pandas as pd
from sklearn.metrics import jaccard_score
finetuned = pd.read_csv("finetuned/membrane-overlay-pos-rel-with-annot-rel-corr-pbr-transmembrane.csv")
pretrained = pd.read_csv("pretrained/membrane-overlay-pos-rel-with-annot-rel-corr-pbr-transmembrane.csv")
shuffled = pd.read_csv("shuffled/membrane-overlay-pos-rel-with-annot-rel-corr-pbr-transmembrane.csv")
print( jaccard_score(finetuned, pretrained) )
print( jaccard_score(finetuned, shuffled) )
print( jaccard_score(pretrained, shuffled) )
