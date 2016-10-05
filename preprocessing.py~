__author__ = 'xywang'
"""
created 04/2016
objective: pre-processes a folder of documents, and outputs a pandas df with term as row and doc as col
pre-processing includes: remove_digits, remove stopwords, remove punctuation, remove special characters, stemming,
                         and tokenize
Besides: (1) can generate term-doc mat with term frequencies
         (2) can generate term-doc mat with tf-idf weights
"""

# ----- for remove headers, quotes and footers in 20Newsgroups files ---

import re

def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')
    return after

_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')

def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)
    """
    good_lines = [line for line in text.split('\n')
                  if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)


def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.
    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text

    # ------------------------


def remove_digits(string):
    import re
    return re.sub("\d+"," ",string)

def remove(f):
    noheader = strip_newsgroup_header(f)
    noquote = strip_newsgroup_quoting(noheader)
    nofooter = strip_newsgroup_footer(noquote)
    res = remove_digits(nofooter)
    return res

class stemming(object):
    """keep only letters by using re, and apply stemming"""
    def __init__(self):
        from nltk.stem import PorterStemmer
        self.porter = PorterStemmer()
    def __call__(self, doc):
        from nltk.tokenize import RegexpTokenizer
        tokenizer = RegexpTokenizer(r'\w+')
        return [self.porter.stem(t) for t in tokenizer.tokenize(doc)]

def fn_tdm_df(docs, xColNames = None, **kwargs):
    ''' create a term document matrix as pandas DataFrame
    with **kwargs you can pass arguments of CountVectorizer
    if xColNames is given the dataframe gets columns Names'''
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer

    # initialize the vectorizer
    vectorizer = CountVectorizer(**kwargs)
    x1 = vectorizer.fit_transform(docs)
    # create dataFrame
    df = pd.DataFrame(x1.toarray().transpose(), index=vectorizer.get_feature_names())
    if xColNames is not None:
        df.columns = xColNames
    return df

def fn_tdm_tfidf(docs, xColNames = None, **kwargs):
    ''' create a term document matrix as pandas DataFrame
    with **kwargs you can pass arguments of CountVectorizer
    if xColNames is given the dataframe gets columns Names'''
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    # initialize the vectorizer
    vectorizer = TfidfVectorizer(**kwargs)
    x1 = vectorizer.fit_transform(docs)
    # create dataFrame
    df = pd.DataFrame(x1.toarray().transpose(), index=vectorizer.get_feature_names())
    if xColNames is not None:
        df.columns = xColNames
    return df


def fn_CorpusFromDIR(xDIR):
    ''' functions to create corpus from a Directories
    Input: Directory
    Output: A dictionary with
             Names of files ['ColNames']
             the text in corpus ['docs']'''
    import os
    import codecs
    Res = dict(docs = [codecs.open(os.path.join(xDIR,f), encoding="latin-1").read() for f in os.listdir(xDIR)],
               ColNames = map(lambda x: x, os.listdir(xDIR)))
    return Res

"""
Previously, docs is opened by "open(os.path.join(xDIR,f)" without using any encoding format, and error occurred.
that's how it is changed to be codecs.open(). Try to change between 'utf-8', 'ascii' and 'latin-1' if error still occurs
"""

# usage :

DIR = '20News_test'
docs = fn_CorpusFromDIR(DIR)['docs']

# if docs are from 20Newsgroup
docs_no_digits = [remove(doc) for doc in docs]
# if docs are not from 20Newsgroup
# docs_no_digits = [remove_digits(doc) for doc in docs]

d1 = fn_tdm_tfidf(docs_no_digits,
                  xColNames=fn_CorpusFromDIR(DIR)['ColNames'],
                  tokenizer=stemming(),
                  stop_words='english',
                  max_df=0.5, min_df=0.2)

# d1 = fn_tdm_tfidf(docs_no_digits, xColNames = fn_CorpusFromDIR(DIR)['ColNames'],
#                tokenizer=None,  stop_words=None, max_df = 0.15, min_df = 0.002)

# d1_t = d1.transpose()
import timeit
start = timeit.default_timer()
d1.to_csv('/home/xywang/Desktop/term_doc.csv', header=False, encoding='ascii')
"""
error occurred on dataframe encoding, so that I specified it to "ascii".
"""
stop = timeit.default_timer()
print stop-start
