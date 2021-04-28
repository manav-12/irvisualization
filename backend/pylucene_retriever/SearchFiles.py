#!/usr/bin/env python

import sys, os, lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import LMDirichletSimilarity
from org.apache.lucene.search.similarities import BM25Similarity

class luceneSearcher:
    def __init__(self,dataflag=''):
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        #print('lucene', lucene.VERSION)
        self.dataflag =dataflag
        directory = SimpleFSDirectory(Paths.get('backend/data/'+self.dataflag +'/files_lucene'))
        self.analyzer = StandardAnalyzer()

    def loadIndexDir(self):
        directory = SimpleFSDirectory(Paths.get('backend/data/'+self.dataflag +'/files_lucene'))
        self.searcher = IndexSearcher(DirectoryReader.open(directory))

    def run(self,query,similarity,k=100,returnScore=False):
        '''
        :param query: user query
        :type query: string
        :param similarity: similarity score algorithm to use example bm25, language model etc
        :type similarity: string
        :param k: Number of elements to retreive
        :type k: int
        :param returnScore: set to true or false whether to return the similarity score or not
        :type returnScore: boolean
        :return: returns the documents that are relevant to the query
        :rtype: list
        '''
        vm_env = lucene.getVMEnv()
        vm_env.attachCurrentThread()
        query = QueryParser("contents", self.analyzer).parse(query)
        if similarity == 'LMDirichlet':
            self.searcher.setSimilarity(LMDirichletSimilarity(mu=2000))
        elif similarity == 'BM25':
            self.searcher.setSimilarity(BM25Similarity())
        scoreDocs = self.searcher.search(query, k).scoreDocs
        relDocs = []
        for scoreDoc in scoreDocs:
            doc = self.searcher.doc(scoreDoc.doc)
            if returnScore ==True:
                relDocs.append((doc.get("name"),scoreDoc.score))
            else:
                relDocs.append(doc.get("name"))
        return relDocs

def initializeRun(query):
    ls = luceneSearcher("")
    ls.loadIndexDir()
    res = ls.run('code optim space effici','BM25',k=10,returnScore=True)
    print(res)
    res = ls.run('code code optim space effici','BM25',k=10,returnScore=True)
    print(res)
    res = ls.run('code^2 optim^1 space^1 effici^1','BM25',k=10,returnScore=True)
    print(res)
    del ls.searcher

if __name__ == '__main__':
    initializeRun('portable operating systems')

