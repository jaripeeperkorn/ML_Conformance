# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:59:44 2021

@author: Jari Peeperkorn
"""


import gensim
from pyemd import emd
from gensim.corpora.dictionary import Dictionary
import numpy as np
from scipy import spatial

from Scripts.Get_Logs import Get_Logs_Embeddings as get_logs

class WmDistance(object):
    def __init__(self, wv, docset1, docset2):
        self.wv = wv
        self.docset1 = docset1
        self.docset2 = docset2
        self.dists = np.full((len(self.docset1), len(self.docset2)), np.nan)
        self.dictionary = Dictionary(documents=self.docset1 + self.docset2)
        self.vocab_len = len(self.dictionary)
        self._cache_nbow()
        self._cache_dmatrix()
    def _cache_nbow(self):
        self.nbow1 = [self._nbow(doc) for doc in self.docset1]
        self.nbow2 = [self._nbow(doc) for doc in self.docset2]
    def _nbow(self, document):
        d = np.zeros(self.vocab_len, dtype=np.double)
        nbow = self.dictionary.doc2bow(document)
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)
        return d
    def _cache_dmatrix(self):
        self.distance_matrix = np.zeros((self.vocab_len, self.vocab_len), dtype=np.double)
        for i, t1 in self.dictionary.items():
            for j, t2 in self.dictionary.items():
                if self.distance_matrix[i, j] != 0.0: continue
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = \
                    spatial.distance.cosine(self.wv[t1], self.wv[t2])
                    #this was changed in latest iteration
                    #np.sqrt(np.sum((self.wv[t1] - self.wv[t2])**2))
    def __getitem__(self, ij):
        if np.isnan(self.dists[ij[0], ij[1]]):
            self.dists[ij[0], ij[1]] = emd(self.nbow1[ij[0]], self.nbow2[ij[1]], self.distance_matrix)
        return self.dists[ij[0], ij[1]]
    

    
def get_variants_list(lst): #get all of the variants in a list, return as list
    st = set(tuple(i) for i in lst) #convert list into set of tuples
    lst2 = list(st) #convert set of tuples into list of tuples
    return [list(e) for e in lst2]

def count_variant(log, variant): #count how many times a variant comes up in list
    c = 0
    for trace in log:
        if trace == variant:
            c += 1
    return(c)

def get_counts(log, variants):
    counts = []
    for var in variants:
        counts.append(count_variant(log, var))
    return counts

def get_variants_and_counts(log):
    variants = get_variants_list(log)
    return variants, get_counts(log, variants)
    
def get_distances(disM, GT_log_counts, pert_log_counts, GT_log_size, pert_log_size):
    precisiontotal = 0.0 
    fitnesstotal = 0.0  
    minima_column = disM.min(axis=0)
    minima_row = disM.min(axis=1)   
    for i in range(0, len(GT_log_counts)):
        fitnesstotal += minima_column[i] * GT_log_counts[i]
    for i in range(0, len(pert_log_counts)):
        precisiontotal += minima_row[i] * pert_log_counts[i]   
    return (1.0 - fitnesstotal/GT_log_size), (1.0 - precisiontotal/pert_log_size)


    
def get_dist(filenamelog, filenamemodel, windowsize,modellogsize=None):
    
    event_log = get_logs.get_event_log(filenamelog)

    
    if modellogsize == None:
        modellogsize = len(event_log)
    
    model_log = get_logs.get_model_log(filenamemodel, logsize=modellogsize , maxtracelength=100, mintracelength=0)

    
    print("Models preprocessed")
    
    event_log_size = len(event_log)
    model_log_size = len(model_log)
    
    model = gensim.models.Word2Vec(event_log + model_log, vector_size= 8, window=windowsize,  min_count=0, sg = 0)
    model.train(event_log + model_log, total_examples=len(event_log + model_log), epochs=300)
    
    print("Gensim model trained")

    event_log_variants, event_log_counts = get_variants_and_counts(event_log)
    model_log_variants, model_log_counts = get_variants_and_counts(model_log)
    
    print("Variants extracted")
    
    WMD = WmDistance(model.wv, model_log_variants, event_log_variants)
    
    def distmatrix(eventlog, modellog):
        distances = np.zeros((len(modellog),len(eventlog)))
        for i in range(len(modellog)):
            for j in range(len(eventlog)):
                distances[i][j] = WMD[i,j]
        return distances
    
    disM = distmatrix(event_log_variants, model_log_variants)
    
    fitness,precision = get_distances(disM, event_log_counts, model_log_counts, event_log_size, model_log_size)
    
    return(precision, fitness)    