import numpy as np

class MetricsUtils():
    def __init__(self):
        pass

    # https://github.com/eXascaleInfolab/pytrec_eval/blob/master/pytrec_eval/metrics.py
    def eval(self,fetched_list, rel_list, question_id):
        rd = len(rel_list[question_id])
        p, r = 0, 0
        if rd > 0 and len(fetched_list) > 0:
            ard = len(fetched_list)
            rel_docs = [x[0] for x in fetched_list]
            num_relevant = len(set(rel_docs).intersection(rel_list[question_id]))
            p = float(num_relevant) / ard
            r = float(num_relevant) / rd
        return (p, r)

    def evalatk(self,fetched_list, rel_list, question_id,k=10):
        fetched_list = fetched_list[:k]
        rd = len(rel_list[question_id])
        p, r = 0, 0
        if rd > 0 and len(fetched_list) > 0:
            ard = len(fetched_list)
            rel_docs = [x[0] for x in fetched_list]
            num_relevant = len(set(rel_docs).intersection(rel_list[question_id]))
            p = float(num_relevant) / ard
            r = float(num_relevant) / rd
        return (p, r)

    def dcg_at_k(self,r, k, method=0):
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')
        return 0.

    def ndcg_at_k(self,r, k, method=0):
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return self.dcg_at_k(r, k, method) / dcg_max

    def calculateNDCG(self, fetched_list, question_id,all_standards_docs, at):
        # List expected documents found in order of relevance
        rel_list_q = dict(all_standards_docs[question_id])
        r = [int(rel_list_q[documents[0]]) if documents[0] in rel_list_q else 0 for documents in fetched_list]
        return self.ndcg_at_k(r, at, method=1)

    def correct_ones_binary(self,fetched_list, rel_list, question_id, at, at_least_score):
        fetched_list = fetched_list[:at]
        rel_list_q = rel_list[question_id]
        rs = [1 if documents[0] in rel_list_q else 0 for documents in fetched_list]
        return rs

    def average_precision(self,r, rel_list):
        r = np.asarray(r)
        out = [np.sum(r[:k + 1] != 0) / (k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.sum(out) / len(rel_list)

    def calculateAvgP(self,fetched_list, rel_list, question_id, at, at_least_score):
        rs = self.correct_ones_binary(fetched_list, rel_list, question_id, at, at_least_score)
        return self.average_precision(rs, rel_list[question_id])

