import math
import operator
from collections import defaultdict

class LMRetrieval():
    def __init__(self,fetch_limit):
        # dictionary to store the document length (key- docID and value- document's length)
        self.dls={}
        self.avdl = self.get_avdl_doclen("backend/indexed_files/Unigram_Tokens.txt")
        # dictionary to store the inverted index
        # (key- tokens and value- dictionary(key- docID and value- term frequency))
        self.inverted_index = self.get_index("backend/indexed_files/Index_Unigram.txt")
        self.dict_unigram_tf = self.create_unigram_tf_dict("backend/indexed_files/Unigram_TF.txt")
        #lambda value used in QueryLikelyhood formula
        self.lmd = 0.35
        self.mu = 2000
        self.fetch_limit = fetch_limit
        self.alpha = 0.6
    # method to read the term frequencies from the file and
    # store it in dict_unigram_tf dictionary
    def create_unigram_tf_dict(self,termFrequency):
        dict_unigram_tf={}
        string_entry=[]
        f = open(termFrequency,'r+')
        lines = f.readlines()
        for line in lines:
            string_entry.append(line.strip())

        for line in string_entry:
            temp = line.split("-->")
            dict_unigram_tf.update({temp[0]:int(temp[1])})
        return dict_unigram_tf


    # method to get the no. of documents in the corpus and
    # the average document length of the corpus
    def get_avdl_doclen(self,docPath):
        totalcount = 0
        f = open(docPath,'r+')
        content = f.read()
        contents = content.split("\n")
        self.N=len(contents)
        for c in contents:
            d = c.split("-->")
            self.dls[d[0]]=d[1]
            totalcount += float(d[1])
        return totalcount/self.N


    # method to read the inverted index from a file and
    # store in it a dictionary with token as the key and
    # doc ID and term frequency in another dictionary as value
    def get_index(self,indexPath):
        inverted_index = {}
        f = open(indexPath,'r+')
        content = f.read()
        contents = content.split("\n")
        for c in contents:
            index = c.split("-->")
            docs = index[1].split(",")
            inlist = {}
            i = 0
            while i <= len(docs) - 1:
                inlist[docs[i][1:]] = docs[i + 1][:-1]
                i = i + 2
            inverted_index[index[0]] = inlist
        return inverted_index

    # method to get relenvant documents for each query by
    # calculating the QueryLikelihood score of each document
    def searching(self,query,feedback_model=None):
        scoredict = {}
        qtermdict = defaultdict(int)
        # for each query term
        query_terms = query.split()
        
        def addterms(term,value):
            if value > 0:
                qtermdict[term] = qtermdict.get(term,0) + value
        [addterms(term,1) for term in query_terms]
        q_len = sum(qtermdict.values())
        [addterms(term,0) for term in feedback_model.keys() if term not in qtermdict.keys()]
        for q,qfi in qtermdict.items():
            if q in self.inverted_index:
                cqi = float(self.dict_unigram_tf[q])
                for docID in self.dls:
                    if docID in self.inverted_index[q]:
                        fi = float(self.inverted_index[q][docID])
                    else:
                        fi = 0.0
                    # method call to the tfidf scoring fucntion
                    score=self.QLDirichlet(docID,fi,cqi)
                    if feedback_model:
                        score = score *((1-self.alpha)*(qfi/q_len)+(self.alpha)*feedback_model.get(q,0))
                    else:
                        score= score * (qfi)
                    #if scoredict.has_key(docID):
                    try:
                        scoredict[docID] = scoredict[docID] + score
                    except KeyError:
                        scoredict[docID] = score
        # sorted list of tuples based on the ranking of each document for each query
        sort_dict = sorted(scoredict.items(), key=operator.itemgetter(1),reverse=True)[:self.fetch_limit]
        return sort_dict

    # method to calculate the document score using the QueryLikelihood formula
    def QLJM(self,dID,fi,cqi):
        score =0
        dl = float(self.dls[dID])
        if float(dl)>0:
            exp1 = (1 - self.lmd) * fi / dl
            exp2 = self.lmd * (cqi / (self.N*self.avdl))
            exp3 = exp1 /exp2
            score = math.log(1 + exp3)
        return score if score > 0 else 0

        # method to calculate the document score using the QueryLikelihood formula
    def QLDirichlet(self,dID,fi,cqi):
        score =0
        dl = float(self.dls[dID])
        lmd = self.mu / (self.mu + float(dl))
        if float(dl)>0:
            exp1 = math.log(1 + fi / (self.mu *  (cqi / (self.N*self.avdl))))
            exp2 = math.log(lmd) 
            score = exp1 + exp2
        return score if score > 0 else 0
