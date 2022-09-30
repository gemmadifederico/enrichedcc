# %%
from pm4py.objects.log.importer.xes import importer as xes_importer
import import_ipynb
from Statsdata import (get_freq_fitness, get_duration_fitness, get_time_fitness, get_activity_freq_stats, get_activity_duration_stats, 
get_freq_hour_normalized, get_freq_position_normalized, get_position_fitness, get_trace_length_stats, get_length_fitness)
from Mining import discovery_inductive, discovery_heuristic
import subprocess
import csv
import sys
import os
import uuid
import pandas as pd
import json
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

# %% [markdown]
# ## Log import

# %%
def execall(path_logA, path_discovered_model):
    logA = xes_importer.apply(path_logA+".xes")
    uuidstr = str(uuid.uuid4())
    PATH = 'Models'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Discovery using Inductive
    print("Discovering using Inductive...")
    pind, filepath_ind = discovery_inductive(logA, uuidstr)
    print("Done")

    # Discovery using Heuristic
    print("Discovering using Heuristic...")
    pheu, filepath_heu = discovery_heuristic(logA, uuidstr)
    print("Done")

    # Discovery of DCR
    # java -jar "dcr-discovery.jar" "path xes" "path JSON .JSON"
    print("Discovering of DCRgraph...")
    filepath_dcr = "Models\dcr_"+uuidstr+".JSON"
    subprocess.call(['java', '-jar', 'dcr-discovery.jar', path_logA+".xes", filepath_dcr])
    print("Done")

    # Discovery of Palia
    # java -jar "palia-discovery.jar" "path xes"
    print("Discovering of Palia graph...")
    filepath_palia = "Models\palia_"+uuidstr+".JSON"
    subprocess.call(['java', '-jar', 'palia-discovery.jar', path_logA+".xes", filepath_palia])
    print("Done")

    # f = open("Models\Discovery_stats_"+uuidstr+".txt", "w+")

    # Discovery of freq stats
    print("Discovering frequency stats...")
    fstats, attrvalues = get_activity_freq_stats(logA)
    # writeTable(fstats, "Frequency", f)
    # fsum, fmean, fmedian, fstdev, fmin, fmax
    print("Done")
    pass

    # Discovery of duration stats
    print("Discovering duration stats...")
    dstats = get_activity_duration_stats(logA)
    if(dstats is None):
        print("No timestamp attribute defined")
    # else: 
        # writeTable(dstats, "Duration", f)
    #  dmean, dmedian, dmin, dmax, dstdev
    print("Done")

    # Discovery of absolute time stats
    print("Discovering time stats...")
    abstime = get_freq_hour_normalized(logA)
    if(abstime is None):
        print("No timestamp attribute defined")
    # else: 
        # writeTable(abstime, "Absolute time", f)
    print("Done")

    # Discovery of freq position stats
    print("Discovering frequency in positions...")
    posfreq = get_freq_position_normalized(logA)
    # writeTable(posfreq, "Position frequency", f)
    print("Done")

    # Discovery of traces length stats
    print("Discovering trace length stats...")
    lenstats = get_trace_length_stats(logA)
    # writeTable(lenstats, "Trace length", f)
    print("Done")


    dicovered_models = open(path_discovered_model + ".json", "w")
    jsondict = {"UUID": uuidstr,"Ind": filepath_ind, "Heu":filepath_heu, "DCR":filepath_dcr, "Palia": filepath_palia, "Freq":fstats, "Dur": dstats, "AbsT": abstime, "Pos":posfreq, "Len":lenstats}
    json.dump(jsondict, dicovered_models)
    dicovered_models.close()

    # f.close()

# %%
if __name__ == "__main__":
    # Discover.bat logs\discovery\discovery-log models\discovered-model
    A = sys.argv[1]
    B = sys.argv[2]
    execall(A,B)
    print("####################")
    print("TASKS COMPLETED")
    print("####################")


