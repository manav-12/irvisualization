import operator
import os
from backend.utils.processData import DataReader

DATA_FLAG = ""


class DocIndexer():
    def __init__(self):
        self.data_path = os.path.join("backend", "data", DATA_FLAG, "data_csvs", "SNOW_" + DATA_FLAG + ".csv")

    def load_data(self):
        dataReader = DataReader()
        return dataReader.read_dataset(self.data_path)

    def create_inverted_index(self):
        index_unigram = {}
        unigram_tokens = {}
        sorted_documents = self.load_data()

        for filename, filecontent in sorted_documents.items():
            doc_id = filename
            doc_text = filecontent.split()
            doc_token = []
            for term in doc_text:
                doc_token.append(term)
                if term in index_unigram:
                    if doc_id not in index_unigram[term]:
                        index_unigram[term].update({doc_id: 1})
                    else:
                        index_unigram[term][doc_id] += 1
                else:
                    index_unigram[term] = {doc_id: 1}
            unigram_tokens[doc_id] = len(doc_token)
        return unigram_tokens, index_unigram

    # to generate the term frequency and document frequency table
    def generate_tf_df(self, unigram_index):
        term_freq = {}
        doc_freq = {}
        for term, document in unigram_index.items():
            term_frequency = 0
            docString = ""
            i = 0
            for doc_id, freq in document.items():
                i += 1
                term_frequency += freq
                docString += doc_id
                if (i < len(document)):
                    docString += " "
            no_of_docs = len(document)
            # dictionary - term frequencies
            term_freq[term] = term_frequency
            # dictionary -  document frequencies
            doc_freq[term] = {docString: no_of_docs}
        return term_freq, doc_freq

    # to sort and write the term frequency table to a file
    def generate_tf_table(self, gram_tf, file_path):
        sort_dict = sorted(sorted(gram_tf.items()), key=operator.itemgetter(1), reverse=True)
        tfstring = ""
        with open(file_path, "w+") as tf_file:
            for term, tf in sort_dict:
                tfstring += str(term) + "-->" + str(tf) + "\n"
            tf_file.write(tfstring.strip())

    # to sort and write document frequency table to file
    def generate_df_table(self, gram_df, file_path):
        sort_dict = sorted(gram_df.items(), key=operator.itemgetter(0))
        with open(file_path, "w+") as df_file:
            write_string = ""
            for term, df in sort_dict:
                df_value = ""
                for doc, freq in df.items():
                    df_value = str(doc) + " -> " + str(freq)
                final_string = str(term) + "-->" + df_value + "\n"
                write_string += final_string
            df_file.write(write_string.strip())

    # to write the inverted index to a file
    def write_index_to_file(self, dict_gram, file_path):
        final_term = ""
        with open(file_path, "w+") as inv_inx_file:
            for k, v in dict_gram:
                value_term = ""
                i = 0
                for key, val in v.items():
                    i += 1
                    value_term += "(" + str(key) + "," + str(val) + ")"
                    if (i < len(v)):
                        value_term += ","

                final_term += str(k) + "-->" + value_term + "\n"
            inv_inx_file.write(final_term.strip())

    # to write the no. of tokens dictionary to file
    def write_tokens_to_file(self, grams_tokens, file_path):
        token_string = ""
        with open(file_path, "w+") as token_file:
            for k, v in grams_tokens.items():
                token_string += str(k) + "-->" + str(v) + "\n"
            # print token_string.strip()
            token_file.write(token_string.strip())


def main_indexer():
    indexer = DocIndexer()
    unigram_tokens, index_unigram = indexer.create_inverted_index()
    # sorted inverted index
    sorted_index_unigram = sorted(index_unigram.items(), key=operator.itemgetter(0))
    parent_path = os.path.join("backend", "data", DATA_FLAG, "indexed_files")
    indexer.write_index_to_file(sorted_index_unigram, os.path.join(parent_path, "Index_Unigram.txt"))
    indexer.write_tokens_to_file(unigram_tokens, os.path.join(parent_path, "Unigram_Tokens.txt"))
    term_freq, doc_freq = indexer.generate_tf_df(index_unigram)
    indexer.generate_tf_table(term_freq, os.path.join(parent_path, "Unigram_TF.txt"))
    indexer.generate_df_table(doc_freq, os.path.join(parent_path, "Unigram_DF.txt"))


if __name__ == '__main__':
    main_indexer()
