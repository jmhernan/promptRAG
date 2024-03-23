import re
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# handle json
def get_metadata_dict(metadata_file):
    metadata_handle = open(metadata_file)
    metadata = json.loads(metadata_handle.read())
    return metadata

# clean text 
def clean_text(text):
    # regex conditions for text cleanup
    COMMAS = re.compile(r',')
    BAD_SYMBOLS_RE = re.compile(r'[\W]')
    REM_LETTER = re.compile(r'(\b\w{1}\b)')
    REM_NAN_WRD = re.compile(r'^(?:\s*nan\s*)')
    #COUNT_NUM = re.compile(r'(\d{3}|\d{2}|\d{1})')
    COVID = re.compile(r'(covid 19|covid-19)')
    COUNT_NUM = re.compile(r'(\b(\d+)\b)')    
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = COMMAS.sub('', text)
    text = REM_LETTER.sub('', text) 
    text = REM_NAN_WRD.sub('', text)
    text = str.lower(text)
    text = COVID.sub('covid_19', text)
    text = COUNT_NUM.sub('COUNT', text)
    return text


def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_similar_words(list_words, top, wb_model):
    list_out = list_words
    for w in wb_model.most_similar(list_words, topn=top):
        list_out.append(w[0])
    return list(set(list_out))

def remove_non_ascii(txt):
    clean_txt = [0]*len(txt)
    for ind, t in enumerate(txt):
        t = t.encode('ascii', 'ignore')
        clean_txt[ind] = t.decode('ascii')
    return clean_txt
 
def remove_empty(txt):
    '''
    Remove empty list entries
    '''
    updated_txt = list(filter(None, txt))
    return updated_txt 

def word_count_entry(txt):
    word_counts = [0] * len(txt)
    for index, obs in enumerate(txt):
        word_counts[index] = len(obs.split())
    return word_counts

def parallelize_list(txt, function, cores=4):
    ls_split = np.array_split(txt, cores)
    pool = Pool(cores)
    arr = np.concatenate(pool.map(function, ls_split))
    pool.close()
    pool.join()
    return arr