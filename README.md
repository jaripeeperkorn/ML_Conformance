
Under construction

Additional material fo the paper "Global Conformance Checking Measures Using Shallow Representation and Deep Learning"

"Script" contains the scripts used to run the methods. 
- The LSTM method can be ran by importing "LSTM_conformance.py" and running the function "get_dist_random(filenamelog, filenamemodel, modellogsize=None, antilogsize=None, modelantilogsize=None, max_occ=3, bidirec=False, n_layers=1, lstmsize=16, dropout=0.0, l1=0.0, l2=0.0, batch_size=64)"
- The WMD, ICT and t2v methods can be used by importing "WMD.py", "ICT.py" or "t2v.py" respectively. And running the function "get_dist(filenamelog, filenamemodel, variants_variant=True, modellogsize=None, max_occ=3, bigram=False, vector_size = None, window_size = 2)"

Data contains the data for the experiments in the paper.
The bigger XES files of the event logs (inluding the ones with noise) can be provided on demand. 

Extra_scripts contain notebooks used to generate the process trees in experiment 2 and add the nosie to the played out model logs, as well as a script used to convert xes to control-flow only csv's (with each line a trace). 

Results contains the full results for experiment 2.
