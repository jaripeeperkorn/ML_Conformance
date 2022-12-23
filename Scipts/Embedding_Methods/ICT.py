#actual function is get_dist


from gensim.corpora.dictionary import Dictionary
import numpy as np

from Scripts.Get_Logs import Get_Logs_Embeddings as get_logs
from Scripts.Get_Logs import Get_Variants as get_variants
from Scripts.Embedding_Methods.Training import Train_Activity_Embeddings as train_embeddings

import math

#p and q are given as nbow, so an array with voc size and count weights
def ACT(p, q, C, k): #for now C is new every trace comparison, ADD LATER old used for the early stopping
    t = 0
    for i in range(0, len(p)):
        pi = p[i] #the weight of the ith element in p trace
        if pi == 0.: #if this activity is not actually in p pi will be zero
            continue
        dummy_s = np.argsort(C[i]) #have to change to only use the thing where q[j] != 0
        s = np.ones(k, dtype=int)
        it = 0
        j = 0
        while it<k and j<len(dummy_s):
            if q[dummy_s[j]] != 0.:
                s[it] = int(dummy_s[j])
                it = it + 1
            j = j+1
        l = 0
        while l<k and pi>0:
            r = min(pi, q[s[l]])
            pi = pi - r
            t = t + r*C[i, s[l]] 
            l = l+1
        if pi != 0:
            t =  t + pi*C[i, s[l-1]]
    return t



class ICT(object):
    def __init__(self, wv, docset1, docset2, k):
        self.wv = wv
        self.docset1 = docset1
        self.docset2 = docset2
        self.k = k
        self.dists = np.full((len(docset1), len(docset2)), np.nan)
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
    def _normalized(self, vec, minima, maxima):
        return (np.divide(vec - minima, maxima - minima))/(np.sqrt(vec.size))     
    def _cache_dmatrix(self):
        self.distance_matrix = np.zeros((self.vocab_len, self.vocab_len), dtype=np.double)
        self.all_vecs = np.array([self.wv[vec] for _,vec in self.dictionary.items()])
        self.minima = self.all_vecs.min(axis=0)
        self.maxima = self.all_vecs.max(axis=0)
        for i, t1 in self.dictionary.items():
            for j, t2 in self.dictionary.items():
                if self.distance_matrix[i, j] != 0.0: continue
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = \
                    np.sqrt(np.sum((self._normalized(self.wv[t1], self.minima, self.maxima) - self._normalized(self.wv[t2], self.minima, self.maxima))**2))
    def __getitem__(self, ij):
        if np.isnan(self.dists[ij[0], ij[1]]):
            self.dists[ij[0], ij[1]] = ACT(self.nbow1[ij[0]], self.nbow2[ij[1]], self.distance_matrix, self.k)
        return self.dists[ij[0], ij[1]]
    
    

def get_distances_vars(disM, event_log_counts, model_log_counts, event_log_size, model_log_size):
    precisiontotal = 0.0 
    fitnesstotal = 0.0  
    #we changed this because random distance is is 0.33
    minima_column = [(3.0 * min((1.0/3.0), disM.min(axis=0)[i])) for i in range(0, len(disM.min(axis=0)))]
    minima_row = [(3.0 * min((1.0/3.0),disM.min(axis=1)[i])) for i in range(0, len(disM.min(axis=1)))]
    for i in range(0, len(event_log_counts)):
        fitnesstotal += minima_column[i] * event_log_counts[i]
    for i in range(0, len(model_log_counts)):
        precisiontotal += minima_row[i] * model_log_counts[i]   
    return (1.0 - fitnesstotal/event_log_size), (1.0 - precisiontotal/model_log_size)


def get_dist_vars(filenamelog, filenamemodel,  modellogsize=None, max_occ=3, bigram=False, vector_size = None, window_size = 2):
    
    if bigram == True:
        event_log = get_logs.get_event_log_bigram(filenamelog, token=False)
    else:
        event_log = get_logs.get_event_log(filenamelog, token=False)

    if modellogsize == None:
        modellogsize = len(event_log)
    
    if bigram == True:
        model_log = get_logs.get_model_log_bigram(filenamemodel, logsize=modellogsize ,
                                                  maxtracelength=100, mintracelength=0, 
                                                  token=False,max_occ=max_occ)
    else:
        model_log = get_logs.get_model_log(filenamemodel, logsize=modellogsize , 
                                           maxtracelength=100, mintracelength=0,
                                           token=False, max_occ=max_occ)
    print("Models preprocessed")
    
    event_log_size = len(event_log)
    model_log_size = len(model_log)
    
    if vector_size == None:
        vector_size = math.ceil(len(get_logs.get_voc(event_log+model_log))**0.5)
        
    model = train_embeddings.train_model(log = event_log + model_log, vector_size = vector_size
                                         , window_size = window_size, min_count = 1, sg = 1, epochs = 500) #we use skip gram because better with low amountsd of data
        
    print("Gensim model trained")
    
    #We only use the variants such that if a log has a high multiplicity of certain variants the model is significantly more efficient
    event_log_variants, event_log_counts = get_variants.get_variants_and_counts(event_log)
    print("got variant 1")
    model_log_variants, model_log_counts, where_zero = get_variants.get_variants_and_counts_and_wherezero(model_log, event_log_variants)
    print("got variant 2")
    where_zero2 = get_variants.get_where_zero(event_log_variants, model_log_variants)
    print("got variants where zero 2")

    ICT_model = ICT(model.wv, model_log_variants, event_log_variants, 3)
    
    def ict_distmatrix(eventlog, modellog, where_zero, where_zero2):
        distances = np.zeros((len(modellog),len(eventlog)))
        k = 0 
        for i in range(len(modellog)):
            l = 0
            if i == where_zero[k]:
                if k < len(where_zero) - 1:
                    k += 1
                for j in range(len(eventlog)):
                    if j != where_zero2[l]:
                        distances[i][j] = ICT_model[i,j]
                    else:
                        if l < len(where_zero2) - 1:
                            l += 1
            else:
                for j in range(len(eventlog)):
                    distances[i][j] = ICT_model[i,j]
        return distances

    def ict_distmatrix_when_no_overlap(eventlog, modellog):
        distances = np.zeros((len(modellog),len(eventlog)))
        for i in range(len(modellog)):
            for j in range(len(eventlog)):
                distances[i][j] = ICT_model[i,j]
        return distances
    
    if len(where_zero) == 0 or len(where_zero2) == 0:
        ict_disM = ict_distmatrix_when_no_overlap(event_log_variants, model_log_variants)
    else:
        ict_disM = ict_distmatrix(event_log_variants, model_log_variants, where_zero, where_zero2)
    print("got matrix")
    fitness_ict, precision_ict = get_distances_vars(ict_disM, event_log_counts, model_log_counts, event_log_size, model_log_size)
    print("got ict distance")
    
    return(fitness_ict, precision_ict) 
  
def get_distances_full(disM, event_log_size, model_log_size):
    precisiontotal = 0.0 
    fitnesstotal = 0.0  
    #we changed this because random distance is is 0.33
    minima_column = [(3.0 * min((1.0/3.0), disM.min(axis=0)[i])) for i in range(0, len(disM.min(axis=0)))]
    minima_row = [(3.0 * min((1.0/3.0),disM.min(axis=1)[i])) for i in range(0, len(disM.min(axis=1)))]
    for i in range(0, event_log_size):
        fitnesstotal += minima_column[i]
    for i in range(0, model_log_size):
        precisiontotal += minima_row[i]  
    return (1.0 - fitnesstotal/event_log_size), (1.0 - precisiontotal/model_log_size)


    
def get_dist_full(filenamelog, filenamemodel,  modellogsize=None, max_occ=3, bigram=False, vector_size = None, window_size = 2):
    
    if bigram == True:
        event_log = get_logs.get_event_log_bigram(filenamelog, token=False)
    else:
        event_log = get_logs.get_event_log(filenamelog, token=False)

    if modellogsize == None:
        modellogsize = len(event_log)
    
    if bigram == True:
        model_log = get_logs.get_model_log_bigram(filenamemodel, logsize=modellogsize ,
                                                  maxtracelength=100, mintracelength=0, 
                                                  token=False,max_occ=max_occ)
    else:
        model_log = get_logs.get_model_log(filenamemodel, logsize=modellogsize , 
                                           maxtracelength=100, mintracelength=0,
                                           token=False, max_occ=max_occ)
    print("Models preprocessed")
    
    event_log_size = len(event_log)
    model_log_size = len(model_log)
    
    if vector_size == None:
        vector_size = math.ceil(len(get_logs.get_voc(event_log+model_log))**0.5)
    
    model = train_embeddings.train_model(log = event_log + model_log, vector_size = vector_size
                                         , window_size = window_size, min_count = 1, sg = 1, epochs = 500) #we use skip gram because better with low amountsd of data
        
    print("Gensim model trained")

    
    ICT_model = ICT(model.wv, model_log, event_log, 3)
    
    def distmatrix(eventlog, modellog):
        distances = np.zeros((len(modellog),len(eventlog)))
        for i in range(len(modellog)):
            for j in range(len(eventlog)):
                distances[i][j] = ICT_model[i,j]
        return distances
    
    disM = distmatrix(event_log, model_log)
    
    fitness,precision = get_distances_full(disM, event_log_size, model_log_size)
    
    return(fitness, precision) 

def get_dist(filenamelog, filenamemodel, variants_variant=True, modellogsize=None, max_occ=3, bigram=False, vector_size = None, window_size = 2):
    if variants_variant==True:
        return  get_dist_vars(filenamelog, filenamemodel,  modellogsize, max_occ, bigram, vector_size, window_size)
    else:
        return get_dist_full(filenamelog, filenamemodel,  modellogsize, max_occ, bigram, vector_size, window_size)