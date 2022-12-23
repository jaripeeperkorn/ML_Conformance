from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from random import choice

from gensim.models.doc2vec import TaggedDocument

def add_tokens(log): #Do you want to add artificial BOS and EOS to every trace
    for i in range(0, len(log)):
        log[i].insert(0, "BOS")
        log[i].append("EOS")
    return log

def delete_nth(order, max_e):
    # Get a new list that we will return
    result = []
    # Get a dictionary to count the occurences
    occurrences = {}
    # Loop through all provided numbers
    for n in order:
        # Get the count of the current number, or assign it to 0
        count = occurrences.setdefault(n, 0)
        # If we reached the max occurence for that number, skip it
        if count >= max_e:
            continue
        # Add the current number to the list
        result.append(n)
        # Increase the 
        occurrences[n] += 1
    # We are done, return the list
    return result

def detect_and_remove_loops(log, max_occ):
    new_log = []
    for trace in log:
        new_log.append(delete_nth(trace, max_occ))
    return new_log


def log_converter(logdummy): #convert pm4py event log to list of lists (activity type)
    outputlog = []
    for i in range(len(logdummy)):
        dummytrace = []
        for j in range(len(logdummy[i])):
            dummytrace.append(logdummy[i][j]['concept:name'].replace(" ", ""))
        outputlog.append(dummytrace)
    return(outputlog)

def get_event_log(filenamelog, token=False):
    event_log = xes_importer.apply(filenamelog)
    event_log = log_converter(event_log) 
    if token == True:
        event_log = add_tokens(event_log)
    return event_log

def get_alphabet(net):
    activities = list({a.label for a in net.transitions if a.label and not '_' in a.label})
    return activities

def get_model_log(filenamemodel, logsize=5000, maxtracelength=100, mintracelength=0, token=False, max_occ = 3):
    net, initial_marking, fm = pnml_importer.apply(filenamemodel)
    simulated_log = simulator.apply(net, initial_marking, variant=simulator.Variants.BASIC_PLAYOUT, parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: logsize})
    simulated_log = log_converter(simulated_log)
    simulated_log = detect_and_remove_loops(simulated_log, max_occ) #could be removed if we do play-out properly
    #append random stuff to trace if we want a minimum length, not recommended!
    if mintracelength != 0:
        voc = get_alphabet(net)
        for i in range(0, len(simulated_log)):
            while len(simulated_log[i]) < mintracelength:
                simulated_log[i].append(choice(voc))
    if token == True:
        simulated_log = add_tokens(simulated_log)
    return simulated_log


def remove_empty(log):
    new_log = []
    for trace in log:
       if len(trace) > 0:
           new_log.append(trace)
    return new_log  

def get_bigram_log(log):
    log = remove_empty(log)
    bigram_log = []
    for i in range(len(log)):
        dummylist = []
        dummylist.append(log[i][0])
        for j in range(1,len(log[i])):
            dummylist.append(log[i][j-1]+log[i][j])
        dummylist.append(log[i][-1])
        bigram_log.append(dummylist)
    return bigram_log


def get_event_log_bigram(filenamelog,token=True):
    onegram_log = get_event_log(filenamelog, token=token)
    return get_bigram_log(onegram_log)
    
def get_model_log_bigram(filenamemodel, logsize=5000, maxtracelength=100, mintracelength=0,token=True, max_occ = 3):
    onegram_log = get_model_log(filenamemodel, logsize=logsize, maxtracelength=maxtracelength, mintracelength=mintracelength, token=token, max_occ=max_occ)
    return get_bigram_log(onegram_log)

 #for trace2vec
 #we'll have to decide how to do this -> because varaints also need tags etc.
 
def add_tags(log):
    taggedlog = []
    for j in range(len(log)):
        ID = str()
        for i in range(len(log[j])):
            ID = ID + log[j][i].replace(" ", "")
        trace_id = [ID]
        td = TaggedDocument(log[j], trace_id)
        taggedlog.append(td)
    return(taggedlog)

def get_voc(log):
    alphabet = set()
    for i in range(len(log)):
        for j in range(len(log[i])):
            alphabet.add(log[i][j])
    return list(alphabet)


