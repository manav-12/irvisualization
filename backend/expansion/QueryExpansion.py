import itertools
import operator
import os
from collections import defaultdict
from nltk import word_tokenize
from sklearn.preprocessing import MinMaxScaler

from backend.utils.TextClean import TextClean
from backend.expansion.ExpansionModel import ExpansionModel
from backend.expansion.Word2Vec import WordEmbeddings
from backend.utils.dataUtils import DataUtils
from backend.utils.processData import DataReader
import numpy as np
VERBOSE = 0

DATA_FLAG = ""

class QueryExpansion:
    def __init__(self):
        self.load_data = DataUtils()
        self.load_initial_data()
        self.stemmed_root, self.wordcount = self.load_data.load_dictionary()
        self.sumFreq = sum(self.wordcount.values())
        self.contextModel = {k: v / self.sumFreq for k, v in self.wordcount.items()}
        self.wed = WordEmbeddings()
        self.wed.loadWordEmbeddings()
        self.expansionModel = ExpansionModel(self.contextModel, self.wed, self.txtClean)

        # max number of final suggested words
        self.number_suggested_words = 10
        # max number added to the query
        self.query_expansion_num = 10

    def set_params_em_model(self, em_model_params):
        if em_model_params:
            self.expansionModel.set_params_em_model(pos_lmd = em_model_params['pos_lmd'], gamma_p = em_model_params['gamma_p'],
                                                    gamma_n = em_model_params['gamma_n'], gamma_c = em_model_params['gamma_c'],
                                                    beta = em_model_params['beta'], n_alpha = em_model_params['n_alpha'])
        else:
            self.expansionModel.set_params_em_model(pos_lmd=0.5, gamma_p=0.2, gamma_n=0.5, gamma_c=0.3, beta=0.2,
                                                    n_alpha=0.2)

    def load_initial_data(self):
        self.txtClean = TextClean()
        self.original_data = DataReader().read_dataset(os.path.join("backend", "data", DATA_FLAG, "data_csvs",
                                                                    "ORI_"+DATA_FLAG+".csv"))
        self.processed_data = DataReader().read_dataset(os.path.join("backend", "data", DATA_FLAG, "data_csvs",
                                                                     "SNOW_"+DATA_FLAG+".csv"))

    def update_query(self, query, fb_answer, w2v_expansion_terms):
        pos_body_text = [self.processed_data[did] for did, score in fb_answer.items() if score == 2]
        neg_body_text = [self.processed_data[did] for did, score in fb_answer.items() if score == 0]
        positive_candidate_term = set(
            [x for doc in pos_body_text for x in word_tokenize(doc) if
             x in self.contextModel.keys() and x.isdigit() == False and len(x) > 1])
        negative_candidate_term = set(
            [x for doc in neg_body_text for x in word_tokenize(doc) if
             x in self.contextModel.keys() and x.isdigit() == False and len(x) > 1])

        positive_term_probablity = self.expansionModel.positiveFeedbackModel(positive_candidate_term,
                                                                             pos_body_text) if positive_candidate_term else defaultdict(
            float)
        negative_term_probablity = self.expansionModel.negativeFeedbackModel(negative_candidate_term,
                                                                                     neg_body_text,
                                                                                     positive_term_probablity) if negative_candidate_term else defaultdict(
            float)

        if len(pos_body_text) == 0:
            print('No Positive Documents')
        if len(neg_body_text) == 0:
            print('No Negative Documents')

        probablity_term_relevance = self.expansionModel.getFinalTermProbability(query, positive_term_probablity,
                                                                                negative_term_probablity,
                                                                                w2v_expansion_terms)
        list_of_et = sorted(probablity_term_relevance.items(), key=operator.itemgetter(1), reverse=True)
        list_of_expansion_terms = [x[0] for x in list_of_et if x[1] > 0 and len(x[0]) > 2
                                   and x[0] not in query.split()][:self.number_suggested_words]
        query_list = query.split()
        query_dict = {t_x: query_list.count(t_x) for t_x in query_list}
        merged_list = list_of_expansion_terms + query_list
        positive_dict = {x: v for x, v in list_of_et if x in merged_list}
        positive_dict = self.expansionModel.normalizeData(
            dict(itertools.islice(positive_dict.items(), self.number_suggested_words)))
        negative_dict = {x: v for x, v in list_of_et if v < 0}
        final_dict = {x: positive_dict.get(x, 0) + query_dict.get(x, 0) + negative_dict.get(x, 0)
                      for x in set(positive_dict).union(query_dict)}

        final_dict = {x:(y*1) if x in query_list else (round(y,4)) for x,y in final_dict.items()}
        # print(list_of_expansion_terms)
        return list_of_expansion_terms, final_dict

    def getQueryResults(self, question_text, feedback):
        '''
        This method is for accessing the query expansion feature through flask from server.py
        :param question_id: query id
        :type question_id:
        :param question_text: query text
        :type question_text: string
        :param feedback: feedback dictionary of postive and negative documents
        :type feedback: dictionary
        :return:
        :rtype:
        '''
        original_question = question_text
        question_text = self.txtClean.normalizeNltkLemma(question_text, 'SNOWSTEM')
        test_question = ''
        suggested_terms = []
        if len(feedback) > 0:
            w2v_expansion_terms = self.expansionModel.centoridWEMethod(original_question)
            suggested_terms, word_weight_dict = self.update_query(question_text, feedback, w2v_expansion_terms)
        print(test_question)

        suggested_terms = [self.stemmed_root[tm] if tm in self.stemmed_root else tm for tm in suggested_terms]
        print(suggested_terms[:10])
        return suggested_terms[:10], word_weight_dict


if __name__ == '__main__':
    queryExpansion = QueryExpansion()
    queryExpansion.getQueryResults("concurrency control mechanisms in operating systems", {'2080': 2, '2541': 2, '3128': 2})
    queryExpansion.getQueryResults("memory management aspects of operating systems", {'1652': 0, '1752': 2, '1951': 0})
    queryExpansion.getQueryResults("memory management aspects of operating systems",
                                   {'1652': 0, '1951': 0})
    queryExpansion.getQueryResults("Fast algorithm for context-free language recognition or parsing",
                                   {'1768': 2, '2112': 0, '2836': 2})
    queryExpansion.getQueryResults("Fast algorithm for context-free language recognition or parsing",
                                   {'2836': 2})
    queryExpansion.getQueryResults( "Fast algorithm for context-free language recognition or parsing",
                                   {'2112': 0})
    queryExpansion.getQueryResults("optimization of intermediate and machine code ",{'1223': 2, '1134': 2, '2586': 2,
                                                                                          '2897': 0})
