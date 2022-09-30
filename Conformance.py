# %%
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.write import write_xes
import pm4py
import import_ipynb
from Statsdata import (get_freq_fitness, get_duration_fitness, get_time_fitness, get_activity_freq_stats, get_activity_duration_stats, 
get_freq_hour_normalized, get_freq_position_normalized, get_position_fitness, get_trace_length_stats, get_length_fitness)
from Mining import discovery_inductive, discovery_heuristic, conformance
import subprocess
import csv
import sys
import os
import uuid
import json
import re
import pandas as pd
from prettytable import PrettyTable as pt


# %%
def writeTable(stats, title, f):
        # Creating object
        tt = pt([""])
        # Adding rows
        for stat, values in stats.items():
                tb = pt()
                rows = []
                if(type(values) != dict):
                        tb.add_row([values])
                else:
                        for act, value in values.items():
                                tb.add_row([act, value])
                tt.add_row([tb.get_string(title=stat, header=False)])
        tt.align[""] = "l"
        
        f.write(tt.get_string(title=title, header=False) + "\n")

# %%
def compute_fitness_values(results, logB, conf):
    results["fitness"] = ""
    logB["pdc:isPos"] = ""
    match conf:
        case 1:  
            # Fair version      
            w_cf = 0.5
            w_df = 0.5

            for index, trace in results.iterrows():
                cf = trace["ccind"]*(w_cf/4) + trace["ccheu"]*(w_cf/4) + trace["ccdcr"]*(w_cf/4) + trace["ccpalia"]*(w_cf/4)
                df = trace["ccfreq"]*(w_df/3) + trace["ccpos"]*(w_df/3) + trace["cclen"]*(w_df/3)
                if((cf+df)>=0.6):
                    logB.loc[logB["case:concept:name"] == trace["id"], "pdc:isPos"] = True
                else: logB.loc[logB["case:concept:name"] == trace["id"], "pdc:isPos"] = False
                results.loc[trace["id"], "fitness"] = cf+df
        case 2:
            # More declarative version
            # ccind	ccheu	ccdcr   ccpalia	ccfreq	ccdur	ccpos	cclen
            w_cf = 0.5
            w_df = 0.5

            for index, trace in results.iterrows():
                cf = trace["ccind"]*(0) + trace["ccheu"]*(0) + trace["ccdcr"]*(w_cf/2) + trace["ccpalia"]*(w_cf/2)
                df = trace["ccfreq"]*(w_df/3) + trace["ccpos"]*(w_df/3) + trace["cclen"]*(w_df/3)
                if((cf+df)>=0.6):
                    logB.loc[logB["case:concept:name"] == trace["id"], "pdc:isPos"] = True
                else: logB.loc[logB["case:concept:name"] == trace["id"], "pdc:isPos"] = False
                results.loc[trace["id"], "fitness"] = cf+df
        case 3:
            # More imperative
            w_cf = 0.5
            w_df = 0.5

            for index, trace in results.iterrows():
                cf = trace["ccind"]*(w_cf/2) + trace["ccheu"]*(w_cf/2) + trace["ccdcr"]*(0) + trace["ccpalia"]*(0)
                df = trace["ccfreq"]*(w_df/3) + trace["ccpos"]*(w_df/3) + trace["cclen"]*(w_df/3)
                if((cf+df)>=0.6):
                    logB.loc[logB["case:concept:name"] == trace["id"], "pdc:isPos"] = True
                else: logB.loc[logB["case:concept:name"] == trace["id"], "pdc:isPos"] = False
                results.loc[trace["id"], "fitness"] = cf+df
        case 4:
            # More control flow
            w_cf = 0.8
            w_df = 0.2

            for index, trace in results.iterrows():
                cf = trace["ccind"]*(w_cf/4) + trace["ccheu"]*(w_cf/4) + trace["ccdcr"]*(w_cf/4) + trace["ccpalia"]*(w_cf/4)
                df = trace["ccfreq"]*(w_df/3) + trace["ccpos"]*(w_df/3) + trace["cclen"]*(w_df/3)
                if((cf+df)>=0.6):
                    logB.loc[logB["case:concept:name"] == trace["id"], "pdc:isPos"] = True
                else: logB.loc[logB["case:concept:name"] == trace["id"], "pdc:isPos"] = False
                results.loc[trace["id"], "fitness"] = cf+df
        case 5:
            # More data flow
            w_cf = 0.2
            w_df = 0.8

            for index, trace in results.iterrows():
                cf = trace["ccind"]*(w_cf/4) + trace["ccheu"]*(w_cf/4) + trace["ccdcr"]*(w_cf/4) + trace["ccpalia"]*(w_cf/4)
                df = trace["ccfreq"]*(w_df/3) + trace["ccpos"]*(w_df/3) + trace["cclen"]*(w_df/3)
                if((cf+df)>=0.6):
                    logB.loc[logB["case:concept:name"] == trace["id"], "pdc:isPos"] = True
                else: logB.loc[logB["case:concept:name"] == trace["id"], "pdc:isPos"] = False
                results.loc[trace["id"], "fitness"] = cf+df
    return results, logB


# %% [markdown]
# ## Core function

# %%
def execall(path_logB, path_logA, path_models, path_output):
    # Opening JSON file
    with open(path_models+".json") as json_file:
        discovered_models = json.load(json_file)
    logA = xes_importer.apply(path_logA+".xes")
    logB = xes_importer.apply(path_logB+".xes")
    uuid = discovered_models["UUID"]
    res = pd.DataFrame()
    PATH = 'Accepted Traces'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Conformance checking (alignment) from Inductive
    print("Conformance checking from Inductive...")
    ccind_traces, ccind = conformance(discovered_models["Ind"], logB)
    ccind_traces.to_csv("Accepted Traces/accepted_traces_ind_"+uuid+".csv", index=False)
    res = res.assign(ccind = ccind_traces.loc[:,"fitness"])
    print("Done")

    # Conformance checking (alignment) from Heuristic
    print("Conformance checking from Heuristic...")
    ccheu_traces, ccheu = conformance(discovered_models["Heu"], logB)
    ccheu_traces.to_csv("Accepted Traces/accepted_traces_heu_"+uuid+".csv", index=False)
    res = res.assign(ccheu = ccheu_traces.loc[:,"fitness"])
    print("Done")

    # Conformance checking of DCR
    # java -jar "dcr-conformance.jar" "path model .JSON" "path logB" open world flag
    print("Conformance of DCR...")
    subprocess.call(['java', '-jar', 'dcr-conformance.jar', discovered_models["DCR"], path_logB+".xes", "FALSE"])

    # The total fitness value is saved in the file dcrcc.txt
    f = open("dcrcc.txt", "r")
    for line in f:
        ccdcr = float(line)
    f.close()

    ccdcr_traces = pd.read_csv("Accepted Traces/accepted_traces_dcr_"+uuid+".csv", names=["id","fitness"])
    res = res.assign(ccdcr = ccdcr_traces.loc[:,"fitness"])
    print("Done")

    # Conformance checking of Palia
    # java -jar "palia-conformance.jar" "path model .JSON" "path logB"
    print("Conformance of Palia...")
    subprocess.call(['java', '-jar', 'palia-conformance.jar', discovered_models["Palia"], path_logB+".xes"])

    # The total fitness value is saved in the file paliacc.txt
    f = open("paliacc.txt", "r")
    for line in f:
        ccpalia = float(line)
    f.close()

    ccpalia_traces = pd.read_csv("Accepted Traces/accepted_traces_palia_"+uuid+".csv", names=["id","fitness"])
    res = res.assign(ccpalia = ccpalia_traces.loc[:,"fitness"])
    print("Done")

    # Conformance of frequency
    print("Conformance of frequency...")
    ccfreq_ev, ccfreq_t, ccfreq = get_freq_fitness(discovered_models["Freq"], logA, logB)
    ccfreq_traces = pd.DataFrame(ccfreq_t.items(), columns=["id", "fitness"])
    res = res.assign(ccfreq = ccfreq_traces.loc[:,"fitness"])
    ccfreq_traces.to_csv("Accepted Traces/accepted_traces_freq_"+uuid+".csv")
    print("Done")

    # Conformance of duration
    if(not discovered_models["Dur"] is None):
        print("Conformance of duration...")
        ccdur_ev, ccdur_t, ccdur = get_duration_fitness(discovered_models["Dur"], logA, logB)
        ccdur_traces = pd.DataFrame(ccdur_t.items(),columns=["id", "fitness"])
        res = res.assign(ccdur = ccdur_traces.loc[:,"fitness"])
        ccdur_traces.to_csv("Accepted Traces/accepted_traces_dur_"+uuid+".csv")
        print("Done")
    else: ccdur = ""

    # Get fitness of absolute time
    if(not discovered_models["AbsT"] is None):
        print("Absolute time comparison...")
        cctime_act, cctime = get_time_fitness(logA, logB)
        print("Done")
    else: cctime = ""

    # Get fitness of events positions
    print("Position frequency comparison...")
    ccpos_t, ccpos = get_position_fitness(discovered_models["Pos"], logB)
    ccpos_traces = pd.DataFrame(ccpos_t.items(),columns=["id", "fitness"])
    res = res.assign(ccpos = ccpos_traces.loc[:,"fitness"])
    ccpos_traces.to_csv("Accepted Traces/accepted_traces_pos_"+uuid+".csv")
    print("Done")

    # Get fitness of trace length
    print("Trace length comparison...")
    cclen_t, cclen = get_length_fitness(discovered_models["Len"], logB)
    cclen_traces = pd.DataFrame(data=cclen_t.items(),columns=["id", "fitness"])
    res = res.assign(cclen = cclen_traces.loc[:,"fitness"])
    res = res.assign(id = cclen_traces.loc[:,"id"])
    cclen_traces.to_csv("Accepted Traces/accepted_traces_len_"+uuid+".csv")
    print("Done")

    res.index += 1 
    other, final_output = compute_fitness_values(res, pm4py.convert_to_dataframe(logB), 1)
    other.to_csv("Results.csv")
    write_xes(pm4py.convert_to_event_log(final_output), path_output+".xes")

    header = ["logA", "logB", "CCHeu", "CCInd", "CCDcr","CCPalia", "CCFreq", "CCDur", "CCTime", "CCPos", "CCLen"]
    values = [path_logA, path_logB, ccheu, ccind, ccdcr, ccpalia, ccfreq, ccdur, cctime, ccpos, cclen]
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


# %%

# Classify.bat logs\test\test-log logs\base\base-log models\discovered-model logs\classified\test-log
if __name__ == "__main__":
    A = sys.argv[1]
    B = sys.argv[2]
    C = sys.argv[3]
    D = sys.argv[4]
    execall(A,B,C,D)
    print("####################")
    print("TASKS COMPLETED")
    print("####################")



