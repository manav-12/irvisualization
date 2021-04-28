import copy
import math

import nltk
import numpy as np
from gensim import matutils
from gensim.models.word2vec_inner import REAL
from nltk import word_tokenize
from collections import defaultdict

from backend.utils.TextClean import STOP_WORDS


class ExpansionModel():
    def __init__(self, contextModel, wed, txtClean):
        self.contextModel = contextModel
        self.wed = wed
        self.txtClean = txtClean

    def set_params_em_model(self,pos_lmd, gamma_p, gamma_n, gamma_c, beta, n_alpha):
        # For ratio of postive feedback and context model
        self.pos_lmd = pos_lmd
        # for ratio of positive feedback model in the negative model calculation
        self.gamma_p = gamma_p
        # for ratio of negative feedback model in the negative model calculation
        self.gamma_n = gamma_n
        # for ratio of context model in the negative model calculation
        self.gamma_c = gamma_c
        # the ratio of negative model to be subtracted from the positive model
        self.beta = beta
        # The ratio of the wordembedding and the pos-neg em model
        self.n_alpha = n_alpha
        # max amount of words before merging the positive,negative and word embedding candidate terms
        self.threshold_words = 20
        # max number of iterations for which em models should run
        self.em_iterations = 20
        # number of word embedding expansion terms
        self.wordem_terms = 50

    def getOccurance(self, term, feedback_text):
        '''
        :param term:
        :type term:
        :param feedback_text:
        :type feedback_text:
        :return:
        :rtype:
        '''
        wordcount = defaultdict(int)
        for word in word_tokenize(feedback_text):
            wordcount[word] += 1
        try:
            return wordcount[term]
        except:
            return 0

    def getAllOccurance(self, feedback_text):
        '''
        :param feedback_text:
        :type feedback_text:
        :return:
        :rtype:
        '''
        wordcount = defaultdict(int)
        for word in feedback_text:
            wordcount[word] += 1
        try:
            return wordcount
        except:
            return None

    def getBooleanOccurance(self, term, feedback_text):
        '''
        :param term: 
        :type term: 
        :param feedback_text: 
        :type feedback_text: 
        :return: 
        :rtype: 
        '''
        wordcount = defaultdict(int)
        for word in word_tokenize(feedback_text):
            wordcount[word] = 1
        try:
            return wordcount[term]
        except:
            return 0

    def multipleNegativeFeedbackModel(self, candidate_term, feedbacks, probability_t_given_pmodel_at_n):
        '''
        :param candidate_term: candidate terms from the negative feedback text
        :type candidate_term: list
        :param feedbacks: negative feedback text
        :type feedbacks: list
        :param probability_t_given_pmodel_at_n: probabilities generate by the positive model
        :type probability_t_given_pmodel_at_n: dict
        :return: terms generated from positive em model with their probablities
        :rtype: dict
        '''
        # To decide when to terminate the iteration
        prev_log_likelihood = -100000
        # randomize the zeroth probability of negative feedback model to 0.5
        probability_t_given_nmodel_at_n = defaultdict(float)
        final_negative_feedback = defaultdict(float)
        for feedbacksText in feedbacks:
            tokenizedDoc = word_tokenize(feedbacksText)
            if len(tokenizedDoc)>0:
                feedbackOccurances = self.getAllOccurance(tokenizedDoc)
                probability_t_given_nmodel_at_n = {term: feedbackOccurances[term] / len(tokenizedDoc) for term in
                                                candidate_term}
                # probability_t_given_nmodel_at_n = {term: np.random.uniform(low=0.0, high=1.0, size=None) for term in
                #                                    candidate_term}
                # probability_t_given_nmodel_at_n[term] =  np.random.uniform(low=0.0, high=1.0, size=None)
                main_prob_nmodel = copy.deepcopy(probability_t_given_nmodel_at_n)
                # Iterations for updating probablity of the negative model while maximizing the log likelihood over the negative feedbacks
                for iter in range(self.em_iterations):
                    # step E
                    # Equation 13
                    def calculate_lnt(term):
                        den_l_n_t = self.gamma_p * probability_t_given_pmodel_at_n.get(term,
                                                                                    0) + self.gamma_n * probability_t_given_nmodel_at_n.get(
                            term, 0) + self.gamma_c * self.contextModel[term]
                        term_lnt = (self.gamma_n * probability_t_given_nmodel_at_n[term]) / den_l_n_t
                        return term_lnt

                    l_n_t = {term: calculate_lnt(term) for term in candidate_term}
                    # step M

                    denominator = sum([feedbackOccurances[term] * l_n_t[term] for term in candidate_term])

                    def eval_negative_prob(term):
                        temp_numerator = feedbackOccurances[term] * l_n_t[term]
                        return temp_numerator / denominator

                    probability_t_given_nmodel_at_n = {term: eval_negative_prob(term) for term in candidate_term}

                    # Calculate Log Likelihood
                    def calculate_log_likelihood(term):
                        log_expr = math.log(
                            self.gamma_p * probability_t_given_pmodel_at_n.get(term, 0) + self.gamma_n *
                            probability_t_given_nmodel_at_n[
                                term] + self.gamma_c * self.contextModel[term])
                        temp_ll = feedbackOccurances[term] * log_expr
                        return temp_ll

                    log_likelihood = sum([calculate_log_likelihood(term) for term in candidate_term])

                    if log_likelihood < prev_log_likelihood:
                        break
                    else:
                        main_prob_nmodel = copy.deepcopy(probability_t_given_nmodel_at_n)
                        prev_log_likelihood = log_likelihood


                # Equation 11
                def calculate_final_fb_neg(term):
                    final_negative_feedback_term = self.gamma_p * probability_t_given_pmodel_at_n.get(term,
                                                                                                    0) + self.gamma_n * \
                                                main_prob_nmodel[term] + self.gamma_c * self.contextModel[term]
                    return final_negative_feedback_term

                final_negative_feedback = {term: max(final_negative_feedback[term], calculate_final_fb_neg(term)) for term
                                        in candidate_term}
        return final_negative_feedback

    def negativeFeedbackModel(self, candidate_term, feedbacks, probability_t_given_pmodel_at_n):
        '''
        :param candidate_term: candidate terms from the negative feedback text
        :type candidate_term: list
        :param feedbacks: negative feedback text
        :type feedbacks: list
        :param probability_t_given_pmodel_at_n: probabilities generate by the positive model
        :type probability_t_given_pmodel_at_n: dict
        :return: terms generated from positive em model with their probablities
        :rtype: dict
        '''
        # To decide when to terminate the iteration
        prev_log_likelihood = -100000
        # randomize the zeroth probability of negative feedback model to 0.5
        probability_t_given_nmodel_at_n = defaultdict(float)

        feedbacksText = ' '.join([fb for fb in feedbacks])
        tokenizedDoc = word_tokenize(feedbacksText)
        feedbackOccurances = self.getAllOccurance(tokenizedDoc)

        probability_t_given_nmodel_at_n = {term: feedbackOccurances[term] / len(tokenizedDoc) for term in
                                           candidate_term}
        # probability_t_given_nmodel_at_n = {term: np.random.uniform(low=0.0, high=1.0, size=None) for term in
        #                                    candidate_term}
        # probability_t_given_nmodel_at_n[term] =  np.random.uniform(low=0.0, high=1.0, size=None)
        main_prob_nmodel = copy.deepcopy(probability_t_given_nmodel_at_n)
        # Iterations for updating probablity of the negative model while maximizing the log likelihood over the negative feedbacks
        for iter in range(self.em_iterations):
            # step E
            # Equation 13
            def calculate_lnt(term):
                den_l_n_t = self.gamma_p * probability_t_given_pmodel_at_n.get(term,
                                                                               0) + self.gamma_n * probability_t_given_nmodel_at_n.get(
                    term, 0) + self.gamma_c * self.contextModel[term]
                term_lnt = (self.gamma_n * probability_t_given_nmodel_at_n[term]) / den_l_n_t
                return term_lnt

            l_n_t = {term: calculate_lnt(term) for term in candidate_term}
            # step M
            denominator = sum([feedbackOccurances[term] * l_n_t[term] for term in candidate_term])
            def eval_negative_prob(term):
                temp_numerator = feedbackOccurances[term] * l_n_t[term]
                return temp_numerator / denominator

            probability_t_given_nmodel_at_n = {term: eval_negative_prob(term) for term in candidate_term}

            # Calculate Log Likelihood
            def calculate_log_likelihood(term):
                log_expr = math.log(
                    self.gamma_p * probability_t_given_pmodel_at_n.get(term, 0) + self.gamma_n *
                    probability_t_given_nmodel_at_n[
                        term] + self.gamma_c * self.contextModel[term])
                temp_ll = feedbackOccurances[term] * log_expr
                return temp_ll

            log_likelihood = sum([calculate_log_likelihood(term) for term in candidate_term])

            if log_likelihood < prev_log_likelihood:
                break
            else:
                main_prob_nmodel = copy.deepcopy(probability_t_given_nmodel_at_n)
                prev_log_likelihood = log_likelihood

        # Equation 11
        def calculate_final_fb_neg(term):
            final_negative_feedback_term = self.gamma_p * probability_t_given_pmodel_at_n.get(term, 0) + self.gamma_n * \
                                           main_prob_nmodel[term] + self.gamma_c * self.contextModel[term]
            return final_negative_feedback_term

        final_negative_feedback = {term: calculate_final_fb_neg(term) for term in candidate_term}
        return final_negative_feedback

    def positiveFeedbackModel(self, candidate_term, feedbacks):
        '''
        :param candidate_term: possible candidate terms from the positive feedback documents
        :type candidate_term: list
        :param feedbacks: feedback document text
        :type feedbacks: list
        :return: terms from positive em model with their probablities
        :rtype: dict
        '''
        feedbacksText = ' '.join([fb for fb in feedbacks])
        tokenizedDoc = word_tokenize(feedbacksText)
        feedbackOccurances = self.getAllOccurance(tokenizedDoc)

        probability_t_given_pmodel_at_n = {term: feedbackOccurances[term] / len(tokenizedDoc) for term in
                                           candidate_term}
        # probability_t_given_pmodel_at_n = {term: np.random.uniform(low=0.0, high=1.0, size=None) for term in
        #                                    candidate_term}

        main_prob_pmodel = copy.deepcopy(probability_t_given_pmodel_at_n)

        # To decide when to terminate the iteration
        prev_log_likelihood = -100000
        # Iterations for updating probablity of the positive model while maximizing the log likelihood over the positive feedbacks
        for iter in range(self.em_iterations):
            # step E
            # Equation 9
            def calculate_hnt(term):
                return (self.pos_lmd * probability_t_given_pmodel_at_n[term]) / (
                        (self.pos_lmd * probability_t_given_pmodel_at_n[term]) + (
                        (1 - self.pos_lmd) * self.contextModel[term]))

            h_n_t = {term: calculate_hnt(term) for term in candidate_term}

            # step M
            denominator = sum([feedbackOccurances[term] * h_n_t[term] for term in candidate_term])

            def eval_positive_prob(term):
                temp_numerator = feedbackOccurances[term] * h_n_t[term]
                return temp_numerator / denominator

            probability_t_given_pmodel_at_n = {term: eval_positive_prob(term) for term in candidate_term}

            # calculate log likelihood
            def calculate_log_likelihood(term):
                temp_log = feedbackOccurances[term] * math.log(
                    self.pos_lmd * probability_t_given_pmodel_at_n[term] + (1 - self.pos_lmd) * self.contextModel[term])
                return temp_log

            log_likelihood = sum([calculate_log_likelihood(term) for term in candidate_term])

            if log_likelihood < prev_log_likelihood:
                break
            else:
                main_prob_pmodel = copy.deepcopy(probability_t_given_pmodel_at_n)
                prev_log_likelihood = log_likelihood

        # Equation 6
        def calculate_final_pos(term):
            temp_final_fb_pos = self.pos_lmd * main_prob_pmodel[term] + ((1 - self.pos_lmd) * self.contextModel[term])
            return temp_final_fb_pos

        final_positive_feedback = {term: calculate_final_pos(term) for term in candidate_term}
        return final_positive_feedback

    def normalizeData(self, rawData):
        try:
            sum_values = sum(rawData.values())
            normData = {x: v / sum_values for x, v in rawData.items()}
        except:
            normData = {}
        return normData

    def getFinalTermProbability(self, query, positive_term_probability, negative_term_probablity, w2v_expansion_terms):
        '''
        :param positive_term_probability: dict of terms and its probablities generated from positive em model
        :type positive_term_probability: dict
        :param negative_term_probablity: dict of terms and its probablities generated from negative em model
        :type negative_term_probablity: dict
        :param w2v_expansion_terms: dict of terms and its probablities generated word2vec model
        :type w2v_expansion_terms: dict
        :return: final probability list with updated probabilities of the terms using an equation
        :rtype: dict
        '''
        # tm_len = np.array([len(positive_term_probability), len(negative_term_probablity),
        #                    self.threshold_words])
        threshold = self.threshold_words

        positive_term_probability = sorted(positive_term_probability.items(), key=lambda item: item[1],
                                           reverse=True)[:threshold]
        positive_term_probability = self.normalizeData(dict(positive_term_probability))
        #print('word_pos:' + str(positive_term_probability))
        w2v_expansion_terms = sorted(w2v_expansion_terms.items(), key=lambda item: item[1], reverse=True)[
                              :threshold]
        w2v_expansion_terms = self.normalizeData(dict(w2v_expansion_terms))
        #print('word_em:'+str(w2v_expansion_terms))
        #print('intersection_terms:'+str(set(positive_term_probability.keys()).intersection(w2v_expansion_terms.keys())))
        all_terms = set(positive_term_probability.keys()).union(w2v_expansion_terms.keys()).union(set(query.split()))

        negative_term_probablity = {x: y for x, y in negative_term_probablity.items() if x in all_terms}
        negative_term_probablity = sorted(negative_term_probablity.items(), key=lambda item: item[1],
                                          reverse=True)[:threshold]
        negative_term_probablity = self.normalizeData(dict(negative_term_probablity))

        def calculate_final_result(term):
            # temp_result = self.n_alpha * (w2v_expansion_terms.get(term, 0)) + (
            #         1 - self.n_alpha) * positive_term_probability.get(term,0) - self.beta * negative_term_probablity.get(
            #     term, 0)
            temp_result = self.n_alpha * (w2v_expansion_terms.get(term, 0)) + \
                          (1-self.n_alpha-self.beta)* positive_term_probability.get(term,0) \
                          - self.beta * negative_term_probablity.get(term, 0)
            return temp_result

        final_result = {term: calculate_final_result(term) for term in all_terms}
        return final_result


    def wordEmbEMFeedbackModel(self, question_text):
        '''
        :param candidate_term: possible candidate terms from the positive feedback documents
        :type candidate_term: list
        :param feedbacks: feedback document text
        :type feedbacks: list
        :return: terms from positive em model with their probablities
        :rtype: dict
        '''
        feedbackOccurances = self.centoridWEMethod(question_text)
        candidate_term = list(feedbackOccurances.keys())

        probability_t_given_wemmodel_at_n = {term: feedbackOccurances[term] / len(candidate_term) for term in
                                           candidate_term}

        main_prob_wemodel = copy.deepcopy(probability_t_given_wemmodel_at_n)

        # To decide when to terminate the iteration
        prev_log_likelihood = -100000
        # Iterations for updating probablity of the positive model while maximizing the log likelihood over the positive feedbacks
        for iter in range(self.em_iterations):
            # step E
            # Equation 9
            def calculate_hnt(term):
                return (self.pos_lmd * probability_t_given_wemmodel_at_n[term]) / (
                        (self.pos_lmd * probability_t_given_wemmodel_at_n[term]) + (
                        (1 - self.pos_lmd) * self.contextModel[term]))

            h_n_t = {term: calculate_hnt(term) for term in candidate_term}

            # step M
            denominator = sum([feedbackOccurances[term] * h_n_t[term] for term in candidate_term])

            def eval_positive_prob(term):
                temp_numerator = feedbackOccurances[term] * h_n_t[term]
                return temp_numerator / denominator

            probability_t_given_wemmodel_at_n = {term: eval_positive_prob(term) for term in candidate_term}

            # calculate log likelihood
            def calculate_log_likelihood(term):
                temp_log = feedbackOccurances[term] * math.log(
                    self.pos_lmd *  probability_t_given_wemmodel_at_n[term] + (1 - self.pos_lmd) * self.contextModel[term])
                return temp_log

            log_likelihood = sum([calculate_log_likelihood(term) for term in candidate_term])

            if log_likelihood < prev_log_likelihood:
                break
            else:
                main_prob_wemodel = copy.deepcopy(probability_t_given_wemmodel_at_n)
                prev_log_likelihood = log_likelihood

        # Equation 6
        def calculate_final_pos(term):
            temp_final_fb_pos = self.pos_lmd * main_prob_wemodel[term] + ((1 - self.pos_lmd) * self.contextModel[term])
            return temp_final_fb_pos

        final_positive_feedback = {term: calculate_final_pos(term) for term in candidate_term}
        return final_positive_feedback

    def centoridWEMethod(self, question_text):
        '''
        :param question_text: user query input
        :type question_text: string
        :return: return the similar terms wrt the query terms and its score
        :rtype: dict
        '''
        lens_query = [word for word in nltk.word_tokenize(question_text) if word not in STOP_WORDS and len(word) > 2]
        stemmed_lens_query = [self.txtClean.snowball.stem(tm) for tm in lens_query]
        word_emb_prob = defaultdict(float)
        if len(lens_query)>1:
            top_similar, qcent_vec = self.wed.centroid_vector(lens_query, self.wordem_terms)
            querylist_sim = [x[0] for x in top_similar if x[0] not in STOP_WORDS and len(x[0]) > 2 and
                             x[0].replace('.','').isdigit()==False]
                             #and self.txtClean.snowball.stem(x[0]) not in stemmed_lens_query]
            for cad in querylist_sim:
                exp_val = 0
                try:
                    cad_vector = matutils.unitvec(self.wed.embedding[cad]).astype(REAL)
                    exp_val = math.exp(np.dot(qcent_vec, cad_vector)/(np.linalg.norm(qcent_vec)*
                                                                                   np.linalg.norm(cad_vector)))
                except:
                    pass
                stemmed_word = self.txtClean.snowball.stem(cad)
                if stemmed_word in self.contextModel:
                    word_emb_prob[stemmed_word] = max(word_emb_prob.get(stemmed_word,0),exp_val)
        else:
            print("problem")
            print(question_text)
        return word_emb_prob

    def fusionBasedMethods(self, question_text):
        '''
        :param question_text: user query input
        :type question_text: string
        :return: return the similar terms wrt the query terms and its score
        :rtype: dict
        '''
        lens_query = [word for word in nltk.word_tokenize(question_text) if word not in STOP_WORDS and len(word) > 2]
        querylist_formed = self.wed.transform(lens_query, self.wordem_terms)
        querylist_formed = [x for x in querylist_formed if x not in STOP_WORDS]
        denominator_prob = defaultdict(float)
        for tkn in lens_query:
            for cad in querylist_formed:
                exp_val = 0
                try:
                    exp_val = math.exp(self.wed.embedding.similarity(tkn, cad))
                except:
                    pass
                denominator_prob[tkn] = denominator_prob[tkn] + exp_val
        sum_prob_term = defaultdict(float)
        max_prob_term = defaultdict(float)
        for candidate in querylist_formed:
            for tkn in lens_query:
                compute = 0
                try:
                    compute = math.exp(self.wed.embedding.similarity(tkn, candidate)) / denominator_prob[tkn]
                except:
                    pass
                sum_prob_term[candidate] = sum_prob_term[candidate] + compute
                max_prob_term[candidate] = max(max_prob_term[candidate], compute)
        max_prob_term = {self.txtClean.snowball.stem(term): score for term, score in max_prob_term.items() if
                         self.txtClean.snowball.stem(term) in self.contextModel.keys()}
        return max_prob_term
