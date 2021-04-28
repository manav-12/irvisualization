import csv
import itertools
import multiprocessing
import re
import string
from time import time
import pandas as pd
import gensim
import nltk
from gensim import matutils
from gensim.models.word2vec_inner import REAL
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import numpy as np
from backend.utils.TextClean import STOP_WORDS, TextClean
import matplotlib.pyplot as plt

class WordEmbeddings():

    def __init__(self):
        pass

    def loadWordEmbeddings(self):
        print('----loading word embeddings-----')
        #model_path = "backend/data/embedding/GoogleNews-vectors-negative300.bin"
        model_path = "backend/data/embedding/ORI_Emb.bin"
        binary = ".bin" in model_path
        self.embedding = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=binary)
        print('----Model Loaded-----')

    def word2VecModelGenerator(self,sentences):
        '''
        :param sentences: list of sentences tokenized
        :type sentences: list
        :return: generates the word embeddings and saves them
        :rtype: N/A
        '''
        cores = multiprocessing.cpu_count()
        t = time()
        w2v_model = Word2Vec(min_count=2, negative=10 ,window=5, sg=1, size=300,workers=cores - 1)
        w2v_model.build_vocab(sentences, progress_per=10000)
        # print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
        # w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=10, report_delay=1)
        # w2v_model.init_sims(replace=True)
        # embedding_overall = gensim.models.KeyedVectors.load_word2vec_format(
        #     "backend/data/embedding/GoogleNews-vectors-negative300.bin", binary=True)
        w2v_model.intersect_word2vec_format("backend/data/embedding/GoogleNews-vectors-negative300.bin", binary=True)
        # w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter, report_delay=1)
        # w2v_model.init_sims(replace=True)
        gensim.models.keyedvectors._save_word2vec_format('backend/data/embedding/ORI_Emb.bin',
                                                         w2v_model.wv.vocab,w2v_model.wv.vectors,binary=True)
        print('model saved')

    def word2VecModelTrainer(self,sentences):
        '''
        :param sentences: list of sentences tokenized
        :type sentences: list
        :return: generates the word embeddings and saves them
        :rtype: N/A
        '''
        cores = multiprocessing.cpu_count()
        t = time()
        w2v_model = Word2Vec(min_count=2,window=3,size=300,negative=10,workers=cores - 1)
        w2v_model.build_vocab(sentences, progress_per=10000)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=5, report_delay=1)
        w2v_model.init_sims(replace=True)
        gensim.models.keyedvectors._save_word2vec_format('backend/data/embedding/ORI_Emb.bin',
                                                         w2v_model.wv.vocab,w2v_model.wv.vectors,binary=True)
        print('model saved')


    def centroid_vector(self, query, top_terms=30):
        query_lst = [word for word in query if word in self.embedding.vocab]
        vectors = np.vstack([self.embedding[word] for word in query_lst]).astype(REAL)
        qcent_vec = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        top_similar = self.embedding.similar_by_vector(qcent_vec, topn=top_terms, restrict_vocab=None)
        return top_similar, qcent_vec

    def transform(self, queryterms,top_terms=10):
        '''
        :param queryterms: terms in the user query
        :type queryterms: list
        :param top_terms: number of top similar terms based on the cosine similarity with the query word
        :type top_terms: integer
        :return: return list of top similar terms
        :rtype: list
        '''
        querylist = []
        for qry in queryterms:
            querylist.extend([qry])
            try:
                querylist.extend(tupword[0].lower() for tupword in self.embedding.similar_by_vector(qry, topn=top_terms))
            except:
                pass
        return querylist

    def generateCorpus(self,stp=True):
        corpus_pd = pd.read_csv("backend/data/analysis_data.csv",";")
        corpus_pd = corpus_pd.fillna("")
        title = corpus_pd.title.tolist()
        content = corpus_pd.abstract.tolist()
        keyword = corpus_pd.keyword.tolist()
        title.extend(content)
        title.extend(keyword)
        sentences = [nltk.sent_tokenize(re.sub('[^A-Za-z0-9.,\'\-\s]+', ' ',articles.lower()))
                     for articles in title if articles != '']
        all_sent = list(itertools.chain(*sentences))
        def remove_elements(tokens):
            return [tkn for tkn in tokens if tkn not in STOP_WORDS]
        if stp:
            corpus = [remove_elements(nltk.word_tokenize(sentence)) for sentence in all_sent]
        else:
            corpus = [nltk.word_tokenize(sentence) for sentence in all_sent]
        return corpus

    def saveCSVVector(self):
        import csv
        with open('backend/data/embedding/wv_embeddings.tsv', 'w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            words = self.embedding.wv.vocab.keys()
            for word in words:
                vector = self.embedding.wv.get_vector(word).tolist()
                row = vector
                writer.writerow(row)

        with open('backend/data/embedding/metadata.tsv', 'w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            words = self.embedding.wv.vocab.keys()
            for word in words:
                row = [word]
                writer.writerow(row)

    def generateVizPlot(self):
        vocab = list(self.embedding.wv.vocab)
        X = self.embedding[vocab]
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)
        df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df['x'], df['y'])
        for word, pos in df.iterrows():
            ax.annotate(word, pos)
        plt.show()

def main():
    wed = WordEmbeddings()
    print('train corpus')
    corpus = wed.generateCorpus(stp=True)
    wed.word2VecModelGenerator(corpus)
    wed.loadWordEmbeddings()
    wed.saveCSVVector()
    return

def test_embeddings():
    wed = WordEmbeddings()
    wed.loadWordEmbeddings()
    while True:
        word = input('similar word for: ')
        print(wed.embedding.most_similar(word))

def testing_centroid():
    wed = WordEmbeddings()
    wed.loadWordEmbeddings()
    while True:
        word = input('similar query for: ')
        top_similar, vector = wed.centroid_vector(word.split())
        print(top_similar)

if __name__ == '__main__':
    main()
