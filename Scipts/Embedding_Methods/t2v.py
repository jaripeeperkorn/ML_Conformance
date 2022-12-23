#actual function is get_dist

import numpy as np
from Scripts.Get_Logs import Get_Logs_Embeddings as get_logs
from Scripts.Get_Logs import Get_Variants as get_variants
from Scripts.Embedding_Methods.Training import Train_Trace_Embeddings as train_embeddings

from scipy import spatial
import math

    

def get_distances_vars(disM, event_log_counts, model_log_counts, event_log_size, model_log_size):
    precisiontotal = 0.0 
    fitnesstotal = 0.0  
    #we changed this because distance can technically be between -1 and 1, almost always positive though
    #everything smaller thn 0 is jsut mapped to zero
    minima_column = [min(1.0, disM.min(axis=0)[i]) for i in range(0, len(disM.min(axis=0)))]
    minima_row = [min(1.0,disM.min(axis=1)[i]) for i in range(0, len(disM.min(axis=1)))]
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
                                                  token=False)
    else:
        model_log = get_logs.get_model_log(filenamemodel, logsize=modellogsize , 
                                           maxtracelength=100, mintracelength=0,
                                           token=False)
    print("Models preprocessed")
    
    
    
    event_log_size = len(event_log)
    model_log_size = len(model_log)
    
    if vector_size == None:
        vector_size = math.ceil(len(get_logs.get_voc(event_log+model_log))**0.5)
    
    train_log = get_logs.add_tags(event_log + model_log) #the log used to train the doc2vec model,he tags are just sentences of all activties in the controlflow appended
        
    model = train_embeddings.train_model(taggedlog = train_log, vector_size = vector_size, window_size = window_size, min_count = 1, dm = 0, epochs = 500) 
    
    print("Gensim model trained")
    
    
    print("Gensim model trained")
    #We only use the variants such that if a log has a high multiplicity of certain variants the model is significantly more efficient
    event_log_variants, event_log_counts = get_variants.get_variants_and_counts(event_log)
    print("got variant 1")
    model_log_variants, model_log_counts, where_zero = get_variants.get_variants_and_counts_and_wherezero(model_log, event_log_variants)
    print("got variant 2")
    where_zero2 = get_variants.get_where_zero(event_log_variants, model_log_variants)
    print("got variants where zero 2")
    
    event_log_variants_tags = get_logs.add_tags(event_log_variants)
    model_log_variants_tags = get_logs.add_tags(model_log_variants)

    def cosdis(trace1, trace2):
        rep1 = model.docvecs[trace1[1]] #I changes this from 0 to 1 since last iteration
        rep2 = model.docvecs[trace2[1]]
        return spatial.distance.cosine(rep1, rep2)

    def distmatrix(eventlog, modellog, where_zero, where_zero2):
        distances = np.zeros((len(modellog),len(eventlog)))
        k = 0 
        for i in range(len(modellog)):
            l = 0
            if i == where_zero[k]:
                if k < len(where_zero) - 1:
                    k += 1
                for j in range(len(eventlog)):
                    if j != where_zero2[l]:
                        distances[i][j] = cosdis(modellog[i], eventlog[j])
                    else:
                        if l < len(where_zero2) - 1:
                            l += 1
            else:
                for j in range(len(eventlog)):
                    distances[i][j] = cosdis(modellog[i], eventlog[j])
        return distances

    def distmatrix_when_no_overlap(eventlog, modellog):
        distances = np.zeros((len(modellog),len(eventlog)))
        for i in range(len(modellog)):
            for j in range(len(eventlog)):
                distances[i][j] = cosdis(modellog[i], eventlog[j])
        return distances

    if len(where_zero) == 0 or len(where_zero2) == 0:
        disM = distmatrix_when_no_overlap(event_log_variants_tags, model_log_variants_tags)
    else:
        disM = distmatrix(event_log_variants_tags, model_log_variants_tags, where_zero, where_zero2)
        
    fitness,precision = get_distances_vars(disM, event_log_counts, model_log_counts, event_log_size, model_log_size)

    print("got t2v distance")
    
    return(fitness, precision)   


def get_distances_full(disM, event_log_size, model_log_size):
    precisiontotal = 0.0 
    fitnesstotal = 0.0  
    #we changed this because distance can technically be between -1 and 1, almost always positive though
    #everything smaller thn 0 is jsut mapped to zero
    minima_column = [min(1.0, disM.min(axis=0)[i]) for i in range(0, len(disM.min(axis=0)))]
    minima_row = [min(1.0,disM.min(axis=1)[i]) for i in range(0, len(disM.min(axis=1)))]
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
                                                  token=False)
    else:
        model_log = get_logs.get_model_log(filenamemodel, logsize=modellogsize , 
                                           maxtracelength=100, mintracelength=0,
                                           token=False)
    print("Models preprocessed")
    
    
    
    event_log_size = len(event_log)
    model_log_size = len(model_log)
    
    if vector_size == None:
        vector_size = math.ceil(len(get_logs.get_voc(event_log+model_log))**0.5)
    
    train_log = get_logs.add_tags(event_log + model_log) #the log used to train the doc2vec model,he tags are just sentences of all activties in the controlflow appended
        
    model = train_embeddings.train_model(taggedlog = train_log, vector_size = vector_size, window_size = window_size, min_count = 1, dm = 0, epochs = 500) 
    
    print("Gensim model trained")
    
    
    event_log_tags = get_logs.add_tags(event_log)
    model_log_tags = get_logs.add_tags(model_log)
    
    print("Variants extracted")
    
    def cosdis(trace1, trace2):
        rep1 = model.docvecs[trace1[1]] #I changes this from 0 to 1 since last iteration
        rep2 = model.docvecs[trace2[1]]
        return spatial.distance.cosine(rep1, rep2)

    def distmatrix(eventlog, modellog):
        distances = np.zeros((len(modellog),len(eventlog)))
        for i in range(len(modellog)):
            for j in range(len(eventlog)):
                distances[i][j] = cosdis(modellog[i], eventlog[j])
        return distances
    
    disM = distmatrix(event_log_tags, model_log_tags)
    
    fitness,precision = get_distances_full(disM, event_log_size, model_log_size)
    
    return(fitness, precision)    

def get_dist(filenamelog, filenamemodel, variants_variant=True, modellogsize=None, max_occ=3, bigram=False, vector_size = None, window_size = 2):
    if variants_variant==True:
        return  get_dist_vars(filenamelog, filenamemodel,  modellogsize, max_occ, bigram, vector_size, window_size)
    else:
        return get_dist_full(filenamelog, filenamemodel,  modellogsize, max_occ, bigram, vector_size, window_size)