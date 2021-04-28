import operator
import pickle

import gensim

from backend.utils.processData import DataReader

vector = pickle.load(open("backend/files_lucene/robust_tfidfModel.pickle", "rb"))
tfidf_doc = pickle.load(open("backend/files_lucene/robust_tfidfMatrix.pickle", "rb"))
vocab = {y: x for x, y in vector.vocabulary_.items()}
with open('backend/files_lucene/DocumentNames.txt', 'r') as f:
    doc_names = f.read().splitlines()
final_dict = {}
for num,doc_names in enumerate(doc_names[:10]):
    imp_keywords = dict(zip([vocab[x] for x in tfidf_doc[num].indices],tfidf_doc[num].data))
    sort_dict = dict(sorted(imp_keywords.items(), key=operator.itemgetter(1), reverse=True)[:15])
    final_dict[doc_names]= ' '.join(sort_dict.keys())
print("success")