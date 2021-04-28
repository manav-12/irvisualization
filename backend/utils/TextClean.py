import re
import unicodedata
import string
import nltk
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer

with open('backend/data/standard_data/common_words', 'r') as f:
    STOP_WORDS = f.read().splitlines()
STOP_WORDS.extend(string.punctuation)
STOP_WORDS.extend(['--'])

class TextClean:
    def __init__(self):
        self.wnl = nltk.WordNetLemmatizer()
        self.porter_stemmer = nltk.PorterStemmer()
        self.snowball = SnowballStemmer("english")


    def reTextNormalization(self, input_text):
        # removing html tags
        temp_text = re.sub(r'<.*?>',' ', input_text)
        # remove non-ascii
        temp_text = re.sub('[^A-Za-z0-9.,\'\-\s]+', ' ', temp_text)
        # remove whitespace
        temp_text = temp_text.strip()
        return temp_text

    def remove_accented_chars(self,text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    #
    # def replace_contractions(self,text):
    #     contraction_re = re.compile('(%s)' % '|'.join(CONTRACTION_MAP.keys()))
    #     def replace(match):
    #         return CONTRACTION_MAP[match.group(0)]
    #     return contraction_re.sub(replace, text)


    def nltk2wn_tag(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def create_ori_stem_map(self, input_text):
        temp_text = input_text.lower()
        temp_text = self.remove_accented_chars(temp_text)
        temp_text = self.reTextNormalization(temp_text)
        ori_map = {self.snowball.stem(word) if word.__contains__('.') == False else word : word
                          for word in nltk.word_tokenize(temp_text) if word not in STOP_WORDS}
        return ori_map

    def normalizeNltkLemma(self, input_text,lemma, lowercase=True):
        if lowercase and lemma != 'RAWREP':
            temp_text = input_text.lower()
            temp_text = self.remove_accented_chars(temp_text)
            temp_text = self.reTextNormalization(temp_text)
        else:
            temp_text = input_text

        if lemma == 'RAWREP':
            final_string = temp_text
        else:
            if lemma == 'RAW':
                sentence_lists = [word for word in nltk.word_tokenize(temp_text) if word not in STOP_WORDS and len(word)>1]
            elif lemma == 'LEMMA':
                nltk_tagged = nltk.pos_tag(nltk.word_tokenize(temp_text))
                wordnet_tagged = map(lambda x: (x[0], self.nltk2wn_tag(x[1])), nltk_tagged)
                sentence_lists = [self.wnl.lemmatize(word.lower(), tag) for word, tag in wordnet_tagged if
                                       word not in STOP_WORDS]
            elif lemma == 'SNOWSTEM':
                sentence_lists = [self.snowball.stem(word) if word.__contains__('.')==False else word
                                  for word in nltk.word_tokenize(temp_text) if word not in STOP_WORDS]
                # and word.lower().islower()
            elif lemma == 'STEM':
                sentence_lists = [self.porter_stemmer.stem(word) if word.__contains__('.')==False else word
                                  for word in nltk.word_tokenize(temp_text) if word not in STOP_WORDS]
            else:
                sentence_lists = [word for word in nltk.word_tokenize(temp_text) if
                                  word not in STOP_WORDS]
            final_string = ' '.join(sentence_lists)
        return final_string


if __name__ == "__main__":
    txt_cls = TextClean()
    txt_cls.create_ori_stem_map("operating operate")
    print(txt_cls.normalizeNltkLemma('Texture analysis by computer.	Digitized texture analysis.  Texture synthesis. Perception of texture.','SNOWSTEM'))