#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pm4py.objects.log.importer.xes import importer as xes_importer
from Statsdata import get_freq_fitness, get_duration_fitness, get_time_fitness
from Mining import discovery_inductive, discovery_heuristic, conformance
import subprocess
import csv
import sys
import os
import uuid
import pandas as pd


# ## Log import

# In[4]:


# Just for thesting
# logB = xes_importer.apply("../logNormal.xes")
# logA = xes_importer.apply("../logFreq.xes")


# In[1]:


def execall(path_logA, path_logB):
    # XES import
    logA = xes_importer.apply(path_logA)
    logB = xes_importer.apply(path_logB)

    # Conformance checking (alignment) from Inductive
    print("Conformance checking from Inductive...")
    ccind_traces, ccind = conformance(logA, logB, "Inductive")
    print("Done")

    # Conformance checking (alignment) from Heuristic
    print("Conformance checking from Heuristic...")
    ccheu_traces, ccheu = conformance(logA, logB, "Heuristic")
    print("Done")

    # Discovery of DCR
    # java -jar "dcr-discovery.jar" "path xes" "path JSON .JSON"
    print("Discovery of DCRgraph...")
    PATH = 'Models'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    filename = "Models\dcr_"+str(uuid.uuid4())+".JSON"
    subprocess.call(['java', '-jar', 'dcr-discovery.jar', path_logA, filename])
    print("Done") 

    # Conformance checking of DCR
    # java -jar "dcr-conformance.jar" "path model .JSON" "path logB" open world flag
    print("Conformance of DCR...")
    subprocess.call(['java', '-jar', 'dcr-conformance.jar', filename, path_logB, "FALSE"])

    # The total fitness value is saved in the file dcrcc.txt
    f = open("dcrcc.txt", "r")
    for line in f:
        ccdcr = float(line)
    f.close()
    print("Done")

    # Conformance of frequency
    print("Conformance of frequency...")
    ccfreq_ev, ccfreq_traces, ccfreq = get_freq_fitness(logA, logB)
    print("Done")

    # Conformance of duration
    print("Conformance of duration...")
    ccdur_ev, ccdur_traces, ccdur = get_duration_fitness(logA, logB)
    print("Done")

    # Get fitness of absolute time
    print("Absolute time comparison...")
    cctime_act, cctime = get_time_fitness(logA, logB)
    print("Done")

    # ccfreq 
    # ccdur 
    # ccheu
    # ccind
    # ccdcr
    # ccdecl

    header = ["logA", "logB", "ccheu", "ccind", "ccdcr", "ccfreq", "ccdur", "cctime"]
    values = [path_logA, path_logB, ccheu, ccind, ccdcr, ccfreq, ccdur, cctime]
    print("Writing the results...")
    # open the file in the write mode

    # Create File
    if not os.path.exists('Results.csv'):
        print("Creating file...")
        with open('Results.csv', 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(values)
            print("Done")
    else:
        with open('Results.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(values)
            print("Done")

    # ccdcr_traces = pd.read_csv('accepted_traces.csv', names=["case_id", "fitness"])
    # Create File
    # print("Creating file...")
    # name = os.path.basename(path_logB)
    # with open("Fitness_"+os.path.splitext(name)[0]+str(uuid.uuid4())[:4]+".csv", 'w', newline="") as f:
        # writer = csv.writer(f)
        # f.write("case_id, ccind, ccheu, ccdcr, ccfreq, ccdur, cctime\n")
        # for trace in logA:
            # key = trace.attributes.get("concept:name")
            # "id, ccind, ccheu, ccdcr, ccfreq, ccdur, cctime"
            # f.write("%s, %s, %s, %s, %s, %s, %s\n" % (key, ccind_traces.loc[ccind_traces["case_id"] == str(key), "fitness"].iloc[0], 
            # ccheu_traces.loc[ccheu_traces["case_id"] == str(key), "fitness"].iloc[0], 
            # ccdcr_traces.loc[ccdcr_traces["case_id"] == 3, "fitness"].iloc[0],
            # ccfreq_traces[key], ccdur_traces[key], cctime_traces[key]))
        # print("Done")


# In[6]:


A = sys.argv[1]
for i in range(2, len(sys.argv)):
    B = sys.argv[i] 
    execall(A,B)
    print("####################")
    print("TASKS COMPLETED")
    print("####################")

