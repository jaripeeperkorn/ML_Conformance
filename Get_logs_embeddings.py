from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from random import choice

def log_converter(logdummy): #convert pm4py event log to list of lists (activity type)
    outputlog = []
    for i in range(len(logdummy)):
        dummytrace = []
        for j in range(len(logdummy[i])):
            dummytrace.append(logdummy[i][j]['concept:name'].replace(" ", ""))
        outputlog.append(dummytrace)
    return(outputlog)

def get_event_log(filenamelog):
    event_log = xes_importer.apply(filenamelog)
    event_log = log_converter(event_log)
    return event_log

def get_alphabet(net):
    activities = list({a.label for a in net.transitions if a.label and not '_' in a.label})
    return activities

def get_model_log(filenamemodel, logsize=5000, maxtracelength=100, mintracelength=0):
    net, initial_marking, fm = pnml_importer.apply(filenamemodel)
    simulated_log = simulator.apply(net, initial_marking, variant=simulator.Variants.BASIC_PLAYOUT, parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: logsize})
    simulated_log = log_converter(simulated_log)
    #append random stuff to trace if we want a minimum length, not recommended!
    if mintracelength != 0:
        voc = get_alphabet(net)
        for i in range(0, len(simulated_log)):
            while len(simulated_log[i]) < mintracelength:
                simulated_log[i].append(choice(voc))
    return simulated_log

def get_bigram_log(log):
    bigram_log = []
    for i in range(len(log)):
        dummylist = []
        for j in range(1,len(log[i])):
            dummylist.append(log[i][j-1]+log[i][j])
        bigram_log.append(dummylist)
    return bigram_log

def get_event_log_bigram(filenamelog):
    onegram_log = get_event_log(filenamelog)
    return get_bigram_log(onegram_log)
    
def get_model_log_bigram(filenamemodel, logsize=5000, maxtracelength=100, mintracelength=0):
    onegram_log = get_model_log(filenamemodel, logsize=logsize, maxtracelength=maxtracelength, mintracelength=mintracelength)
    return get_bigram_log(onegram_log)