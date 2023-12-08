# %%
from numpy import NaN
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.write import write_xes
import pm4py
import import_ipynb
from Statsdata import (get_freq_fitness, get_duration_fitness, get_time_fitness, get_activity_freq_stats, get_activity_duration_stats, 
get_freq_hour_normalized, get_freq_position_normalized, get_position_fitness, get_trace_length_stats, get_length_fitness)
from Mining import discovery_inductive, discovery_heuristic, conformance, conformance_declare
import subprocess
import csv
import sys
import os
import uuid
import json
import re
import pandas as pd
from prettytable import PrettyTable as pt
from statistics import mean 


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
def compute_fitness_values(ccheu, ccind, ccdcr, ccfreq, ccdur, cctime, cclen, conf):
    result = 0
    match conf:
        case 1:  
            # Fair version      
            w_cf = 0.5
            w_df = 0.5
            # In the control flow we consider ind, heu and dcr as 0.5/3 each
            cf = ccind*(w_cf/3) + ccheu*(w_cf/3) + ccdcr*(w_cf/3)
            # In the data flow we consider freq, dur, time, len as 0.5/4 each  (ccfreq, ccdur, cctime, cclen)
            df = ccfreq*(w_df/4) + ccdur*(w_df/4) + cctime*(w_df/4) + cclen*(w_df/4)
            result = cf+df
        case 2:  
            cf_el = pd.Series([ccheu, ccind, ccdcr])
            df_el = pd.Series([ccfreq, ccdur, cctime, cclen])
            w_cf = 0.5
            w_df = 0.5
            # In the control flow we consider ind, heu and dcr as 0.5/3 each
            cf = cf_el.mean()
            df = df_el.mean()
            result = (cf*w_cf)+(df*w_df)
    return result, cf, df


# %% [markdown]
# ## Core function

# %%
def execall(path_logB, path_logA, path_models, case, exp_name):
    # Test log, base log, models path
    # Opening JSON file
    with open(path_models+"models.json") as json_file:
        discovered_models = json.load(json_file)
    logA = xes_importer.apply(path_logA+".xes")
    logB = xes_importer.apply(path_logB+".xes")
    uuid = discovered_models["UUID"]
    res = pd.DataFrame()
    PATH = os.path.join(path_models, 'Accepted Traces')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Conformance checking (alignment) from Inductive
    print("Conformance checking from Inductive...")
    ccind_traces, ccind, precind = conformance(discovered_models["Ind"], logB)
    # ccind_traces.to_csv(PATH+"/accepted_traces_ind_"+uuid+".csv", index=False)
    res = res.assign(ccind = ccind_traces.loc[:,"fitness"])
    print("Done")

    # # Conformance checking (alignment) from Heuristic
    # print("Conformance checking from Heuristic...")
    # ccheu_traces, ccheu = conformance(discovered_models["Heu"], logB)
    # ccheu_traces.to_csv(PATH+"/accepted_traces_heu_"+uuid+".csv", index=False)
    # # res = res.assign(ccheu = ccheu_traces.loc[:,"fitness"])
    # print("Done")
    ccheu = None

    # Conformance checking of DCR
    # java -jar "dcr-conformance.jar" "path model .JSON" "path logB" open world flag
    print("Conformance of DCR...")
    subprocess.call(['java', '-jar', 'dcr-conformance.jar', discovered_models["DCR"], path_logB+".xes", "TRUE"])
    
    # Conformance checking of Declare
    print("Conformance checking of Declare...")
    ccdeclare = conformance_declare(discovered_models["Declare"], path_logB+".xes")
    # ccdeclare = NaN
    print("Done")

    # The total fitness value is saved in the file dcrcc.txt
    f = open("dcrcc.txt", "r")
    for line in f:
        ccdcr = float(line)
    f.close()

    # ccdcr_traces = pd.read_csv(PATH+"/accepted_traces_dcr_"+uuid+".csv", names=["id","fitness"])
    # res = res.assign(ccdcr = ccdcr_traces.loc[:,"fitness"])
    print("Done")

    # # Conformance checking of Palia
    # # java -jar "palia-conformance.jar" "path model .JSON" "path logB"
    # print("Conformance of Palia...")
    # subprocess.call(['java', '-jar', 'palia-conformance.jar', discovered_models["Palia"], path_logB+".xes"])

    # # The total fitness value is saved in the file paliacc.txt
    # f = open("paliacc.txt", "r")
    # for line in f:
    #     ccpalia = float(line)
    # f.close()

    # ccpalia_traces = pd.read_csv("Accepted Traces/accepted_traces_palia_"+uuid+".csv", names=["id","fitness"])
    # res = res.assign(ccpalia = ccpalia_traces.loc[:,"fitness"])
    # print("Done")

    # Conformance of frequency
    print("Conformance of frequency...")
    ccfreq_ev, ccfreq_t, ccfreq = get_freq_fitness(discovered_models["Freq"], logA, logB)
    ccfreq_traces = pd.DataFrame(ccfreq_t.items(), columns=["id", "fitness"])
    res = res.assign(ccfreq = ccfreq_traces.loc[:,"fitness"])
    # ccfreq_traces.to_csv(PATH+"/accepted_traces_freq_"+uuid+".csv")
    print("Done")

    # Conformance of duration
    if(not discovered_models["Dur"] is None):
        print("Conformance of duration...")
        ccdur_ev, ccdur_t, ccdur = get_duration_fitness(discovered_models["Dur"], logA, logB)
        ccdur_traces = pd.DataFrame(ccdur_t.items(),columns=["id", "fitness"])
        res = res.assign(ccdur = ccdur_traces.loc[:,"fitness"])
        # ccdur_traces.to_csv(PATH+"/accepted_traces_dur_"+uuid+".csv")
        print("Done")
    else: ccdur = ""

    # Get fitness of absolute time
    if(not discovered_models["AbsT"] is None):
        print("Absolute time comparison...")
        cctime_act, cctime = get_time_fitness(logA, logB)
        print("Done")
    else: cctime = ""

    # # Get fitness of events positions
    # print("Position frequency comparison...")
    # ccpos_t, ccpos = get_position_fitness(discovered_models["Pos"], logB)
    # ccpos_traces = pd.DataFrame(ccpos_t.items(),columns=["id", "fitness"])
    # res = res.assign(ccpos = ccpos_traces.loc[:,"fitness"])
    # ccpos_traces.to_csv("Accepted Traces/accepted_traces_pos_"+uuid+".csv")
    # print("Done")

    # Get fitness of trace length
    print("Trace length comparison...")
    cclen_t, cclen = get_length_fitness(discovered_models["Len"], logB)
    cclen_traces = pd.DataFrame(data=cclen_t.items(),columns=["id", "fitness"])
    res = res.assign(cclen = cclen_traces.loc[:,"fitness"])
    # res = res.assign(id = cclen_traces.loc[:,"id"])
    # cclen_traces.to_csv(PATH+"/accepted_traces_len_"+uuid+".csv")
    print("Done")

    res.index += 1 
    # tot_fitness, cf_fitness, df_fitness = compute_fitness_values(ccheu, ccind, ccdcr, ccfreq, ccdur, cctime, cclen, 2)
    # other.to_csv("Results.csv")
    # write_xes(pm4py.convert_to_event_log(final_output), path_output+".xes")

    # # Get the timestamp of the last trace of the testing log, to obtain the week number
    loglen = len(logB)
    lasttracelen = len(logB[loglen-1])
    timest = logB[loglen-1][lasttracelen-1].get("start_timestamp")

    header = ["Case", "Conf", "logA", "logB", "CCHeu", "CCInd", "PrecInd", "CCDcr", "CCDeclare", "CCFreq", "CCDur", "CCTime", "CCLen", "Timestamp"]
    values = [case, '\"'+exp_name+'\"', path_logA, path_logB, ccheu, ccind, precind, ccdcr, ccdeclare, ccfreq, ccdur, cctime, cclen, timest]
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
    C = sys.argv[3] + "/" 
    D = sys.argv[4]
    E = sys.argv[5]
    execall(A,B,C,D,E)
    print("####################")
    print("TASKS COMPLETED")
    print("####################")