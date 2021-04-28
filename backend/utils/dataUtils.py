import csv
import os
from collections import defaultdict

from nltk import word_tokenize
csv.field_size_limit(10000000)
DATA_FLAG = ""


class DataUtils:
    def __init__(self):
        pass

    def all_scores(self, f_path):
        file = open(f_path, mode="r", encoding='utf-8', errors='ignore').read()
        files = [x.split() for x in file.split('\n')]
        rel_list = {}
        for x in files:
            rel_list[x[0]] = []
        for i in files:
            rel_list[i[0]].append((i[2], i[3]))
        # print rel_list
        return rel_list

    def document_with_original_score(self, question_id, rel_list):
        return {docs[0]: docs[1] for docs in rel_list[question_id]}

    def gold_standard(self, f_path):
        """
        This methods generates the gold standards that is the actual relevant documents to the query
        :param f_path: path of the trec relevance file
        :type f_path: file
        :return: return a dictionary with keys as the query id and the values will be the list of relevant documents
        :rtype:  dictionary
        """
        file = open(f_path, mode="r", encoding='utf-8', errors='ignore').read()
        files = [x.split() for x in file.split('\n')]
        # print files
        rel_list = {}
        for x in files:
            rel_list[x[0]] = []
        for i in files:
            if int(i[3]) != 0:
                rel_list[i[0]].append(i[2])
        # print rel_list
        return rel_list

    def fetch_queries(self, f_path):
        """
        :param f_path: file path for fetching the queries
        :type f_path: file
        :return: dictionary of query id and the query text
        :rtype: dictionary
        """
        file = open(f_path, mode="r", encoding='utf-8', errors='ignore').read()
        query = [x.split(' ', 1) for x in file.split('\n')]
        query_dict = {}
        for q in query:
            query_dict[q[0]] = q[1].strip()
        return query_dict

    def build_query(self, test_question):
        return test_question.title + ' ' + test_question.body

    def load_dictionary(self, min_freq=2):
        """
        :param min_freq: count of the word to decide if the word should be included in the dictonary or not
        :type min_freq:int
        :return:stemmed_root- dict with stem word and its actual word, wordcount- dict with frequency of the word
        :rtype: dict,dict
        """
        wordcount = defaultdict(int)
        stemmed_root = defaultdict(str)
        id_cnt = 1

        with open(os.path.join("backend", "data", DATA_FLAG, "data_csvs", "countDict.txt")) as f:
            for line in f:
                term, freq, doc_freq = line.split()
                freq = int(freq)
                doc_freq = int(doc_freq)
                if doc_freq >= min_freq:
                    wordcount[term] = freq
                    id_cnt += 1

        with open(os.path.join("backend", "data", DATA_FLAG, "data_csvs", "stem_map.txt")) as f:
            for line in f:
                stem_word, actual_word = line.split()
                stemmed_root[stem_word] = actual_word
        print(str(id_cnt) + ' terms have been loaded to the dictionary with the minimum frequency of ' + str(min_freq))
        return stemmed_root, wordcount

    def load_data_csv(self, file_path):
        """
        :param file_path: file path of the csv with doc id and text
        :type file_path: file
        :return: dict of doc id as key and doc text as value
        :rtype: dict
        """
        with open(file_path, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=';')
            next(reader)
            sorted_doc = {}
            for row in reader:
                sorted_doc[row[0]] = row[1]
        return sorted_doc

    def generate_occurance(self):
        """
        :return: This method processes csv containing the document id and text and generates the word freq
        :rtype: N/A
        """
        from collections import defaultdict
        wordcount = defaultdict(int)
        document_count = defaultdict(int)
        with open(os.path.join("backend", "data", DATA_FLAG, "data_csvs", "SNOW_"+DATA_FLAG+".csv"), 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=';')
            next(reader)
            sorted_doc = {}
            for row in reader:
                sorted_doc[row[0]] = row[1]
        doc_list = [doc for k, doc in sorted_doc.items()]

        for x in doc_list:
            for word in word_tokenize(x):
                wordcount[word] += 1
            for word in set(word_tokenize(x)):
                document_count[word] += 1

        with open(os.path.join("backend", "data", DATA_FLAG, "data_csvs", "countDict.txt"), 'w') as f:
            f.writelines('{} {} {}\n'.format(k, v, document_count[k]) for k, v in wordcount.items())

        print('generated')


if __name__ == '__main__':
    load_data = DataUtils()
    load_data.generate_occurance()
