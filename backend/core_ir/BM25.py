import math
import operator


class BM25Retrieval():
    def __init__(self, fetch_limit):
        # dictionary to store the inverted index
        # (key- tokens and value- dictionary(key- docID and value- term frequency))
        # path to the unigram tokens(docuemnt and the no. of unigrams it contains) file
        # list to hold the ranked documents
        self.fetch_limit = fetch_limit
        # dictionary to store the document length (key- docID and value- document's length)
        self.dls = {}
        self.avdl = self.get_avdl_doclen("backend/indexed_files/Unigram_Tokens.txt")
        self.inverted_index = self.get_index("backend/indexed_files/Index_Unigram.txt")

    # method to get the no. of documents in the corpus and
    # the average document length of the corpus
    def get_avdl_doclen(self, docPath):
        totalcount = 0
        f = open(docPath, 'r+')
        content = f.read()
        contents = content.split("\n")
        self.N = len(contents)
        for c in contents:
            d = c.split("-->")
            self.dls[d[0]] = d[1]
            totalcount += float(d[1])
        return totalcount / self.N

    # method to read the inverted index from a file and
    # store in it a dictionary with token as the key and
    # doc ID and term frequency in another dictionary as value
    def get_index(self, indexPath):
        inverted_index = {}
        f = open(indexPath, 'r+')
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
    # calculating the tfidf score of each document
    def searching(self, qterms):
        # for every query
        scoredict = {}
        qtermdict = {}
        # qterms = word_tokenize(self.txtClean(query))
        # for each query term
        for q in qterms.split():
            try:
                qtermdict[q] = qtermdict[q] + 1
            except KeyError:
                qtermdict[q] = 1

        for q, qfi in qtermdict.items():
            if q in self.inverted_index:
                for docID in self.inverted_index[q]:
                    fi = float(self.inverted_index[q][docID])
                    score = self.BM25(docID, len(self.inverted_index[q]), fi, qfi)
                    # if scoredict.has_key(docID):
                    try:
                        scoredict[docID] = scoredict[docID] + score
                    except KeyError:
                        scoredict[docID] = score

        # sorted list of tuples based on the ranking of each document for each query
        sort_dict = sorted(scoredict.items(), key=operator.itemgetter(1), reverse=True)[:self.fetch_limit]
        return sort_dict

    # method to calculate the document score using the BM25 formula
    def BM25(self, dID, ni, fi, qfi):
        k1 = 1.2
        b = 0.75
        k2 = 100
        dl = float(self.dls[dID])
        kval = k1 * ((1 - b) + (b * (dl / self.avdl)))
        p1 = math.log(1+((self.N - ni + 0.5) / (ni + 0.5)))
        p2 = ((k1 + 1) * fi) / (kval + fi)
        p3 = (((float(k2) + 1) * float(qfi)) / (float(k2) + float(qfi)))
        score = p1 * p2 * p3
        return score
