from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = 'backend/data/embedding/glove.6B.300d.txt'
tmp_file = "backend/data/embedding/glovec.6B.300d.txt"

_ = glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)