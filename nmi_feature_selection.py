__author__ = 'xywang'

"""
created 04/2016
objective: given d documents of c classes, select n terms of highest nmi values for each class
Can be used to select features of a corpus.

references:
theory: http://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html
source: http://stackoverflow.com/questions/15899861/efficient-term-document-matrix-with-nltk
count nonzeros: http://stackoverflow.com/questions/3797158/counting-non-zero-elements-within-each-row-and-within-each-column-of-a-2d-numpy

Note: in the folder, need to remove *.txt~ files if exist, check the class_doc_dict={}
"""

import preprocessing

DIR = 'docs'
docs = preprocessing.fn_CorpusFromDIR(DIR)['docs']
docs_no_digits = [preprocessing.remove_digits(doc) for doc in docs]

d1 = preprocessing.fn_tdm_df(docs_no_digits, xColNames = preprocessing.fn_CorpusFromDIR(DIR)['ColNames'],
               tokenizer = preprocessing.stemming(),
               stop_words='english')

d1.to_csv('output.txt')

columns = (d1 != 0).sum(0)
rows = (d1 != 0).sum(1)

def get_terms_in_a_column(df, col):
    col_df = df.loc[df[col] >= 1]
    terms_set = set(col_df.index.tolist())
    return terms_set

col_names = list(d1.columns.values)
doc_term_dict = {col_name:get_terms_in_a_column(d1,col_name) for col_name in col_names}
class_doc_dict = {1:{"doc1.txt", "doc2.txt", "doc3.txt"},2:{"doc4.txt", "doc5.txt"}}

def generate_class_term_dict(doc_term_dict,class_doc_dict):
    class_term_dict = dict()
    for cla in class_doc_dict:
        doc_list = class_doc_dict[cla]
        term_set = set()
        for doc in doc_list:
            term_set.update(doc_term_dict[doc])
        class_term_dict[cla] = term_set
    return class_term_dict

class_term_dict = generate_class_term_dict(doc_term_dict, class_doc_dict)

def get_N11(df, term, clas):
    matched_flag = (df.ix[term] > 0)
    col_names = df.columns
    matched_cols = col_names[matched_flag == True]
    set_doc_has_term = set(matched_cols.tolist())
    set_doc_in_class = class_doc_dict[clas]
    intersection = set_doc_has_term.intersection(set_doc_in_class)
    return len(intersection)

def compute_NMI(df, term, clas):
    import numpy as np
    import math
    e = 0.00001
    N = len(df.columns) # num of total docs
    cols = df.loc[[term]]
    N1_ = float(np.count_nonzero(cols.values)) # num of docs that contain term
    N0_ = float(N - N1_)
    N11 = float(get_N11(df, term, clas)) # num of docs that contain term and in class
    N10 = float(N1_ - N11)
    N_1 = float(len(class_doc_dict[clas])) # num of docs that in class
    N01 = float(N_1 - N11)
    N00 = float(N0_ - N01)
    N_0 = float(N10 + N00)
    t1 = N11/N * math.log(N*N11/(N1_*N_1+e)+e, 2)
    t2 = N01/N * math.log(N*N01/(N0_*N_1+e)+e, 2)
    t3 = N10/N * math.log(N*N10/(N1_*N_0+e)+e, 2)
    t4 = N00/N * math.log(N*N00/(N0_*N_0+e)+e, 2)
    NMI = t1 + t2 + t3 + t4
    return NMI

results = dict()

for clas in class_doc_dict.keys():
    results[clas] = list()
    for term in list(d1.index):
        results[clas].append((term, compute_NMI(d1, term, clas)))


def select_n_top_terms_for_each_class(results):
    from operator import itemgetter
    res = dict()
    for clas in results:
        sorting_by_nmi = sorted(results[clas], key=itemgetter(1), reverse=True)
        terms_with_top_nmi = [sorting_by_nmi[i][0] for i in range(10)]
        res[clas] = terms_with_top_nmi
    return res

print select_n_top_terms_for_each_class(results)
