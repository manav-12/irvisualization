import copy
import math
import operator
import os
import pickle
from collections import defaultdict
import nltk
import pandas as pd
from sklearn.manifold import TSNE
from backend.pylucene_retriever.UpdateIndex import LuceneIndexUpdater
from backend.utils.TextClean import TextClean
from backend.utils.dataUtils import DataUtils
from backend.utils.metricsUtils import MetricsUtils
from backend.utils.processData import DataReader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from backend.pylucene_retriever.SearchFiles import luceneSearcher
import scipy
import logging

PERPLEXITY = 30
# logging.basicConfig(filename='backend/logs/vizmatrixlog_runF3c1.log',
#                      level=logging.DEBUG)
DATA_FLAG = ""


class VizMatrix:
    def __init__(self, orig_docs, all_documents, fetch_limit):
        # dictionary to store the inverted index
        # (key- tokens and value- dictionary(key- docID and value- term frequency))
        # path to the unigram tokens(docuemnt and the no. of unigrams it contains) file
        # list to hold the ranked documents
        self.fetch_limit = fetch_limit
        # dictionary to store the document length (key- docID and value- document's length)
        self.dls = {}
        self.txt_cls = TextClean()
        self.avdl = self.get_avdl_doclen("backend/data/" + DATA_FLAG + "/indexed_files/Unigram_Tokens.txt")
        self.inverted_index = self.get_index("backend/data/" + DATA_FLAG + "/indexed_files/Index_Unigram.txt")
        self.orig_docs = orig_docs
        self.tokinized_orig_docs = {x : nltk.word_tokenize(v.lower()) for x,v in self.orig_docs.items()}
        self.all_docs = all_documents
        self.txt_cls = TextClean()
        self.ls = luceneSearcher(DATA_FLAG)
        self.load_data = DataUtils()
        self.eval_metrics = MetricsUtils()
        parent_path = os.path.join("backend", "data", DATA_FLAG)
        self.short_queries = self.load_data.fetch_queries(
            os.path.join(parent_path, "standard_data", "Topics.txt"))
        # this would return all the long query - dercription based on the titles
        self.long_queries = self.load_data.fetch_queries(
            os.path.join(parent_path, "standard_data", "TopicDesc.txt"))
        # this is the truth table of just the documents that have score 1 or 2
        self.gold_std = self.load_data.gold_standard(
            os.path.join(parent_path, "standard_data", "qrels.txt"))
        # this gives all the documents that are linked to a query with score 0,1,2
        self.all_standards_docs = self.load_data.all_scores(
            os.path.join(parent_path, "standard_data", "qrels.txt"))
        self.vector = pickle.load(open(os.path.join(parent_path, "data_csvs", "tfidfModel.pickle"), "rb"))
        self.tfidf_doc = pickle.load(open(os.path.join(parent_path, "data_csvs", "tfidfMatrix.pickle"), "rb"))
        self.vocab = {y: x for x, y in self.vector.vocabulary_.items()}
        logging.debug('--queries and goldstandards fetched--')
        with open(os.path.join(parent_path, "data_csvs", "DocumentNames.txt"), 'r') as f:
            doc_names = f.read().splitlines()
        self.doc_names = doc_names
        self.final_dict = {}
        # stemFile = open(os.path.join("backend", "data", DATA_FLAG, "data_csvs", "stem_highlight.pkl"), "rb")
        # self.highlight_map = pickle.load(stemFile)
        self.tfidf_dict={}
        for num, doc_names in enumerate(doc_names):
            imp_keywords = dict(zip([self.vocab[x] for x in self.tfidf_doc[num].indices], self.tfidf_doc[num].data))
            self.tfidf_dict[doc_names] = imp_keywords
            sort_dict = dict(sorted(imp_keywords.items(), key=operator.itemgetter(1), reverse=True))
            self.final_dict[doc_names] = list(sort_dict.keys())[:36]
        print("loaded")

    # method to get the no. of documents in the corpus and
    # the average document length of the corpus
    def get_avdl_doclen(self, doc_path):
        totalcount = 0
        f = open(doc_path, 'r+')
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
    def get_index(self, index_path):
        inverted_index = {}
        f = open(index_path, 'r+')
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

    def extend_list_inline(self,parent_list,added_word):
        parent_list.extend([added_word])
        return parent_list

    def get_root_word(self,stem_word,orig_query):
        #word_list_overall =  self.highlight_map[stem_word]
        try:
            document_word =  orig_query[stem_word]
        except:
            document_word = stem_word
        return document_word

    def get_terms_matched(self, q_ori_terms_map, qterms):
        curr_list = []
        doc_words = defaultdict(list)
        for term in qterms:
            try:
                doc_list_ii = list(self.inverted_index[term].keys())
                curr_list.extend(doc_list_ii)
                doc_words = {doc: list(set(self.extend_list_inline(doc_words.get(doc,[]),self.get_root_word(term,q_ori_terms_map))
                if doc in doc_list_ii else doc_words.get(doc,[]))) for doc in self.doc_names}
            except:
                logging.warning("word not found")
                continue
        set_of_combinations = [list(item) for item in set(tuple(row) for row in list(doc_words.values()))]
        set_of_combinations.sort(key=len, reverse=True)
        doc_terms_count = {doc_name: curr_list.count(doc_name) for doc_name in set(curr_list)}
        doc_words_group_map = {doc_name: set_of_combinations.index(wrd_lst) for doc_name, wrd_lst in doc_words.items()}
        doc_terms_count['Query'] = len(qterms)
        #doc_words = {doc_name: curr_list.count(doc_name) for doc_name in set(curr_list)}
        return doc_terms_count, doc_words_group_map, set_of_combinations

    # method to calculate the document score using the tfidf formula
    def tfidf(self,termf,dID):
        fi = termf[dID]
        ni = len(termf)
        dl=self.dls[dID]
        tf=float(fi)/float(dl)
        idf=1+ math.log((self.N+1)/(ni+1))
        score = tf*idf
        return score


    # method to get relenvant documents for each query by
    # calculating the tfidf score of each document
    def searching_lucene(self, doc_terms, qterms, doc_id, expansionTerms, expansionMap, limit=3500):
        if doc_id =="Query":
            if expansionTerms:
                new_term_list = doc_terms+expansionTerms
                doc_terms_weight = ' '.join([x+'^'+str(expansionMap.get(x,1)) for x in new_term_list])
            else:
                doc_terms_weight = ' '.join(doc_terms)
        else:
            if expansionTerms:
                new_term_list = qterms + expansionTerms
                doc_terms_weight = ' '.join([x+'^'+str((expansionMap.get(x,1)+float(self.tfidf_dict[doc_id].get(x,0))))
                                             if doc_terms in new_term_list
                                         else x+'^'+str(float(self.tfidf_dict[doc_id].get(x,0))) for x in doc_terms])
            else:
                doc_terms_weight = ' '.join(
                    [x + '^' + str((qterms.count(doc_terms)+float(self.tfidf_dict[doc_id].get(x,0)))) if doc_terms in qterms
                     else x + '^' + str(float(self.tfidf_dict[doc_id].get(x,0))) for x in doc_terms])
            # doc_terms_weight = ' '.join([x + '^1'  if doc_terms in qterms
            #                              else x + '^0.5' for x in doc_terms])
        # len_doc = sum([1 if x in qterms else 0.5 for x in doc_terms])
        # print(doc_terms_weight)
        if doc_terms_weight.strip() != '':
            score_dict = self.ls.run(doc_terms_weight, similarity='BM25', k=limit, returnScore=True)
            #score_dict = [(tp_d[0],tp_d[1]/len_doc) for tp_d in score_dict]
        else:
            score_dict = {}
        return dict(score_dict)

    def retreival_score_feature(self, sym_matrix):
        query = sym_matrix.loc['Query'].to_numpy().reshape(1, -1)
        new_sym_matrix = sym_matrix.iloc[:-1]
        cosine_score = cosine_similarity(new_sym_matrix, query).reshape(-1)
        retrieval_score = dict(zip(list(new_sym_matrix.index), list(cosine_score)))
        retrieval_score = {k: v for k, v in sorted(retrieval_score.items(), key=lambda item: item[1], reverse=True)}
        return retrieval_score

    # replace with euclidiean
    def retreival_score(self, doc_rep_df):
        """
        :param doc_rep_df: TSNE representation of documents - pandas dataframe
        :return: dict with document name as keys and euclidean distance as values wrt query in sorted order
        """
        # Fetch query vector from the dataframe
        query = doc_rep_df[doc_rep_df['name'] == 'Query'].iloc[:, 1:].to_numpy().reshape(1, -1)
        # include all documents except query vector
        data_doc = doc_rep_df[doc_rep_df['name'] != 'Query']
        new_sym_matrix = data_doc.iloc[:, 1:]
        # euclidean distance of the documents wrt query
        euc_dist = scipy.spatial.distance.cdist(new_sym_matrix, query, metric='euclidean').reshape(-1)
        retrieval_score = dict(zip(list(data_doc['name']), list(euc_dist)))
        retrieval_score = {k: v for k, v in sorted(retrieval_score.items(), key=lambda item: item[1], reverse=False)}
        return retrieval_score

    def bm25baseline(self, query_tup):
        q_terms = self.txt_cls.normalizeNltkLemma(query_tup[1], "SNOWSTEM")
        self.ls.loadIndexDir()
        query_res = dict(self.ls.run(q_terms, similarity='BM25', k=3500, returnScore=True))
        rel_docs_base = list(query_res.items())
        precision20base, recall20base = self.eval_metrics.evalatk(rel_docs_base, self.gold_std, query_tup[0], 20)
        ndcgbase = self.eval_metrics.calculateNDCG(rel_docs_base, query_tup[0], self.all_standards_docs, 20)
        map = self.eval_metrics.calculateAvgP(rel_docs_base, self.gold_std, query_tup[0], 20, 1)
        logging.debug('precision20 for bm25 is: %f' % precision20base)
        logging.debug('recall20 for bm25 is: %f' % recall20base)
        logging.debug('ndcg20 for bm25 is: %f' % ndcgbase)
        logging.debug('map20 for bm25 is: %f' % map)
        metric_result = {'precision20': precision20base, 'recall20': recall20base, 'ndcg': ndcgbase,
                         'curr_question': query_tup[1]}
        return None, rel_docs_base, metric_result

    def create_matrix(self, query_tup, expansionTerms, expansionMap, feedback_docs=None):
        multilevel_dict = {}
        if query_tup[1].strip() == '':
            return "", "", ""
        else:
            q_ori_terms_map = self.txt_cls.create_ori_stem_map(query_tup[1])
            q_terms = self.txt_cls.normalizeNltkLemma(query_tup[1], "SNOWSTEM")
            # word_prob = self.evaluator.expansionModel.fusionBasedMethods(qterms)
            qterms_list = q_terms.split()
            ls_update = LuceneIndexUpdater()
            ls_update.update_lucene_q_idx('Query', q_terms)
            self.ls.loadIndexDir()
            num_matched,  doc_words_group_map, set_of_combinations = self.get_terms_matched(q_ori_terms_map, qterms_list)
            terms_only_doc = [doc for doc, num in num_matched.items() if num > 0]
            query_res = self.searching_lucene(qterms_list, qterms_list,'Query',expansionTerms, expansionMap)
            doc_fr_matrix = copy.deepcopy(terms_only_doc)
            doc_fr_matrix.append('Query')
            for curr_doc in doc_fr_matrix:
                if curr_doc.strip() != 'Query':
                    terms_doc_filtered =  list(set([wrd for wrd in self.all_docs[curr_doc].split()
                        if wrd in qterms_list or wrd in self.final_dict[curr_doc]]))
                    multilevel_dict[curr_doc] = self.searching_lucene(terms_doc_filtered, qterms_list,curr_doc
                                                                      ,expansionTerms, expansionMap)
                else:
                    multilevel_dict[curr_doc] = query_res
            # Creates a matrix of BM25 score of the documents wrt other documents
            final_matrix = pd.DataFrame.from_dict(multilevel_dict, orient='index')
            final_matrix = final_matrix[final_matrix.index]
            # sorting rows and column index in an sync order
            final_matrix = final_matrix.reindex(sorted(final_matrix.columns), axis=1).sort_index(axis=0)
            # divide every element of the row with the diagonal element
            final_matrix = final_matrix.div(np.diag(final_matrix), axis=0)
            # replace inf with nan and fill nan with 0
            final_matrix = final_matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # create a symmetric matrix by adding the dij and dji element and dividing it by 2 and updating both the
            # elements by the average
            arr = final_matrix.to_numpy()
            arr2 = (arr + arr.T) / 2
            # arr2 = np.maximum(arr,arr.T)
            sym_matrix = pd.DataFrame(arr2, index=final_matrix.index.tolist(), columns=final_matrix.columns)
            # replace the elements with the dissimilarity score
            sym_matrix = (2 / (1 + sym_matrix)) - 1
            sym_matrix = sym_matrix.clip(lower=0)
            max_element = sym_matrix.stack().max()
            if feedback_docs:
                pos_docs = [did for did, score in feedback_docs.items() if score == 2]
                neg_docs = [did for did, score in feedback_docs.items() if score == 0]
                sym_matrix.at[pos_docs, 'Query'] = 0.0
                sym_matrix.at['Query', pos_docs] = 0.0
                sym_matrix.at[neg_docs, 'Query'] = max_element
                sym_matrix.at['Query', neg_docs] = max_element

            data_tsne = TSNE(random_state=42, perplexity=PERPLEXITY,n_jobs=-1).fit_transform(sym_matrix)
            doc_rep_df = pd.DataFrame({'name': sym_matrix.index.tolist(), 'X1': data_tsne[:, 0], 'X2': data_tsne[:, 1]})
            try:
                retreival_scores = self.retreival_score(doc_rep_df)
            except:
                logging.warning("error occurred")
                pass
            doc_rep_df['matchKey'] = doc_rep_df['name'].map(num_matched).fillna(0).astype(int)
            doc_rep_df['word_group'] = doc_rep_df['name'].map(doc_words_group_map).fillna(len(set_of_combinations)).\
                astype(int)
            scatterplot_json = doc_rep_df.to_json(orient='records')
            rel_docs = list(retreival_scores.items())
            precision20, recall20 = self.eval_metrics.evalatk(rel_docs, self.gold_std, query_tup[0], 20)
            ndcg = self.eval_metrics.calculateNDCG(rel_docs, query_tup[0], self.all_standards_docs, 20)
            map = self.eval_metrics.calculateAvgP(rel_docs, self.gold_std, query_tup[0], 20, 1)
            logging.debug('precision20 for viz is: %f' % precision20)
            logging.debug('recall20 for viz is: %f' % recall20)
            logging.debug('ndcg20 for viz is: %f' % ndcg)
            logging.debug('map20 for viz is: %f' % map)
            set_of_combinations = {x:v for x,v in enumerate(set_of_combinations)}
            metric_result = {'precision20': precision20, 'recall20': recall20, 'ndcg': ndcg,
                             'curr_question': query_tup[1]}
            # distance learning - put weights on pair of distance
            # metric learning
            ls_update.delete_q_from_idx()
            print("request completed..")
            query_res.pop('Query', None)
            return scatterplot_json, set_of_combinations, rel_docs, metric_result


if __name__ == '__main__':
    dataReader = DataReader()
    ori_doc = dataReader.read_dataset("backend/data/" + DATA_FLAG + "/data_csvs/ORI_" + DATA_FLAG + ".csv")
    all_doc = dataReader.read_dataset("backend/data/" + DATA_FLAG + "/data_csvs/SNOW_" + DATA_FLAG + ".csv")
    retrieval = VizMatrix(ori_doc, all_doc, 3500)
    query_map = retrieval.short_queries
    q_ls = ['22', '31']
    p=[]
    r=[]
    ndcg=[]
    for x in query_map.keys():
        try:
            logging.debug("=======================================" + x
                          + "=======================================")
            logging.debug("BASE APPROACH")
            doc_rep_df, retreival_scores, metric_result = retrieval.bm25baseline((x, query_map[x]))
            logging.debug("VISUALIZATION APPROACH")
            doc_rep_df, set_of_combinations, retreival_scores, metric_result = \
                retrieval.create_matrix((x, query_map[x]),expansionTerms=[],expansionMap={})
            p.append(metric_result['precision20'])
            r.append(metric_result['recall20'])
            ndcg.append(metric_result['ndcg'])
        except:
            print("exception occurred")
    logging.debug(sum(p)/len(p))
    logging.debug(sum(r)/len(r))
    logging.debug(sum(ndcg)/len(ndcg))