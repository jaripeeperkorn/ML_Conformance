from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from random import choice
import random
import copy

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

def add_tokens(log): #Do you want to add artificial BOS and EOS to every trace
    for i in range(0, len(log)):
        log[i].insert(0, "BOS")
        log[i].append("EOS")
    return log

def get_event_log(filenamelog, token=False):
    event_log = xes_importer.apply(filenamelog)
    event_log = log_converter(event_log) 
    if token == True:
        event_log = add_tokens(event_log)
    return event_log

def get_alphabet(net):
    activities = list({a.label for a in net.transitions if a.label and not '_' in a.label})
    return activities

def get_model_log(filenamemodel, logsize=5000, maxtracelength=100, mintracelength=0, token=False, max_occ=3):
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



def get_voc(log):
    outputset1 = set([])
    for trace in log:
        for act in trace:
            outputset1.add(act)
    voc = list(outputset1)
    return voc

def delete_not_new(log, antilog):
    unique = []
    for trace in log:
        if trace not in unique:
            unique.append(trace)
        else:
            continue
    #print(unique)
    new_antilog = []
    for trace in antilog:
        if trace not in unique:
            new_antilog.append(trace)
        else:
            continue
    return(new_antilog)

def get_antilog_random(log, voc, log_size=None, delete_correct=False):
    min_length = len(min(log, key=len))
    max_length = len(max(log, key=len)) #now we just take random between min and max, should probably be changed to normal distribution
    if log_size == None:
        log_size = len(log) #for size antilog = size log, but can be altered
    antilog = []
    for i in range(0, log_size):
        size = random.randint(min_length,max_length)
        antitrace = []
        for j in range(0, size):
            index = random.randint(0,len(voc)-1)
            antitrace.append(voc[index])
        antilog.append(antitrace)
    if delete_correct == True:
        antilog = delete_not_new(log, antilog)
    return(antilog)

def add_noise_name(logdummy, voc):
    changelog = []
    for i in range(0, len(logdummy)):
        trace = copy.deepcopy(logdummy[i])
        random_act = random.randint(0, len(logdummy[i]) - 1)
        new = random.choice(voc)
        #print(new)
        trace[random_act] = new
        changelog.append(trace)
    return changelog

def add_noise_order(logdummy):
    changelog = []
    for i in range(0, len(logdummy)):
        trace = copy.deepcopy(logdummy[i])
        random_act = random.randint(0, len(logdummy[i]) - 1)
        random_act2 = random.randint(0, len(logdummy[i]) - 1)
        first = copy.copy(logdummy[i][random_act])
        second = copy.copy(logdummy[i][random_act2])
        #print(first, second)
        trace[random_act] = second
        trace[random_act2] = first
        changelog.append(trace)
    return changelog

def get_antilog_noise(log, voc, delete_correct=False):
    voc = get_voc(log)
    l = copy.deepcopy(log)
    l1 = add_noise_name(l, voc)
    antilog = add_noise_order(l1)
    if delete_correct == True:
        antilog = delete_not_new(log, antilog)
    return(antilog)