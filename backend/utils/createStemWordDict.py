import os
import pickle
import re

from nltk import word_tokenize, SnowballStemmer
from collections import defaultdict

from backend.utils.TextClean import STOP_WORDS, TextClean
from backend.utils.processData import DataReader
DATA_FLAG = ""

txt_cls = TextClean()
def pre_process(temp_text):
    temp_text = txt_cls.remove_accented_chars(temp_text)
    temp_text = txt_cls.reTextNormalization(temp_text)
    return temp_text

def main():
    '''
    generate a dictionary with stem word as it key and the root word as value
    '''
    ss = SnowballStemmer("english")
    wordcount = defaultdict(list)
    parent_path = os.path.join("backend", "data", DATA_FLAG, "data_csvs")
    original_data = DataReader().read_dataset(os.path.join(parent_path, "ORI_"+DATA_FLAG+".csv"))
    doc_list = [pre_process(doc.lower()) for k, doc in original_data.items()]
    for x in doc_list:
        for word in word_tokenize(x):
            if len(word) > 1 and word not in STOP_WORDS:
                print(word)
                wordcount[ss.stem(word)].append(word.lower())
    word_count_final = {k: list(set(v)) for k, v in wordcount.items()}

    with open(os.path.join(parent_path, "stem_map.txt"), 'w') as f:
        f.writelines('{} {}\n'.format(k, v[0]) for k, v in wordcount.items())

    stem_file = open(os.path.join(parent_path, "stem_highlight.pkl"), "wb")
    pickle.dump(word_count_final, stem_file)
    stem_file.close()


if __name__ == '__main__':
    main()
