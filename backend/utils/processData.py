import csv
import os
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from backend.utils.TextClean import TextClean
from yellowbrick.text import TSNEVisualizer
csv.field_size_limit(10000000)
DATA_FLAG = ""


class DataReader:

    def __init__(self):
        self.txt_cls = TextClean()

    def read_medline(self, originalText=False):
        doc_list = {}
        orig_doc = {}
        path = "backend\\data\\Medline\\text_files"
        for subdir, dirs, files in os.walk(path):
            for file in files:
                file_path = subdir + os.path.sep + file
                doc_text = re.sub('\r\n|\r', ' ', open(file_path, 'r').read())
                orig_doc[file] = doc_text
                doc_list[file] = self.txt_cls.normalizeNltkLemma(doc_text, 'SNOWSTEM')
        if not originalText:
            return doc_list
        else:
            return orig_doc, doc_list

    def read_dataset(self, file_path):
        with open(file_path, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=';')
            next(reader)
            sorted_doc = {}
            for row in reader:
                sorted_doc[row[0]] = row[1]
        return sorted_doc

    def vectorize(self, doc_list):
        doc_text = [doc for k, doc in doc_list.items()]
        doc_names = [k for k, doc in doc_list.items()]
        parent_path = os.path.join("backend", "data", DATA_FLAG, "data_csvs")
        with open(os.path.join(parent_path, "DocumentNames.txt"), mode='wt') as myfile:
            myfile.write('\n'.join(doc_names))
        vectorizer = TfidfVectorizer(lowercase=True, max_df=0.8, min_df=1, stop_words=None)
        fit_vectorizer = vectorizer.fit(doc_text)
        transformed_vectorizer = vectorizer.transform(doc_text)
        with open(os.path.join(parent_path, "tfidfModel.pickle"), 'wb') as fin:
            pickle.dump(fit_vectorizer, fin)
        with open(os.path.join(parent_path, "tfidfMatrix.pickle"), 'wb') as fin:
            pickle.dump(transformed_vectorizer, fin)
        tsne = TSNEVisualizer()
        target = [1 for x in doc_names]
        tsne.fit(transformed_vectorizer, target)
        tsne.show()
        return fit_vectorizer, transformed_vectorizer


if __name__ == '__main__':
    dataReader = DataReader()
    data_path = os.path.join("backend", "data", DATA_FLAG, "data_csvs", "SNOW_" + DATA_FLAG + ".csv")
    doc_list = dataReader.read_dataset(data_path)
    dataReader.vectorize(doc_list)
