# %%
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import pm4py
import uuid
import os
import pandas as pd
from declare4py.declare4py import Declare4Py
from statistics import mean



# %% [markdown]
# ## DISCOVERY INDUCTIVE AND HEURISTIC

# %%
def discovery_inductive(log, uuidstr, path):
    # ind = pm4py.discover_petri_net_inductive(log, noise_threshold=0.9)
    ind = pm4py.discover_petri_net_inductive(log)
    gviz = pn_visualizer.apply(ind[0], ind[1], ind[2])
    filename = path+"/ind_"+uuidstr
    pn_visualizer.save(gviz, filename+".png")
    pm4py.write_pnml(ind[0], ind[1], ind[2], filename+".pnml")
    return ind, filename
def discovery_heuristic(log, uuidstr, path):
    heu = pm4py.discover_petri_net_heuristics(log)
    gviz = pn_visualizer.apply(heu[0], heu[1], heu[2])
    filename = path+"/heu_"+uuidstr
    pn_visualizer.save(gviz, filename+".png")
    pm4py.write_pnml(heu[0], heu[1], heu[2], filename + ".pnml")
    return heu, filename
def discovery_declare(log, uuidstr, path):
    filename = path+"/declare_"+uuidstr
    d4py = Declare4Py()
    d4py.parse_xes_log(log)
    d4py.compute_frequent_itemsets(min_support=0.9)
    d4py.discovery(consider_vacuity=False, max_declare_cardinality=1, output_path=filename+".decl")
    return filename
# def discovery_powl(log, uuidstr, path):
#     powl = pm4py.discover_powl(log)
#     gviz = pn_visualizer.apply(powl[0], powl[1], powl[2])
#     filename = path+"/powl_"+uuidstr
#     pn_visualizer.save(gviz, filename+".png")
#     pm4py.write_pnml(powl[0], powl[1], powl[2], filename+".pnml")
#     return powl, filename


# %% [markdown]
# ## CONFORMANCE - ALIGNMENT

# %%
def conformance_old (logA, logB, alg):
    if(alg == "Inductive"):
        pn = discovery_inductive(logA)
    elif(alg == "Heuristic"):
        pn = discovery_heuristic(logA)
    else:
        raise Exception("Discovery algorithm not recognized")

    aligned_traces = alignments.apply_log(logB, pn[0], pn[1], pn[2])
    x = 0
    for trace in aligned_traces:
        x =  x + trace["fitness"]
    x = int(x)

    aligned_traces_dataframe = alignments.get_diagnostics_dataframe(logB, aligned_traces)
    return aligned_traces_dataframe, x/len(logB)

# %%
def conformance (model, logB):
    net, initial_marking, final_marking = pm4py.read_pnml(model+".pnml")
    soundness = pm4py.objects.petri_net.utils.check_soundness.check_easy_soundness_net_in_fin_marking(net, initial_marking, final_marking)
    if soundness is False:
        aligned_traces_dataframe = pd.DataFrame()
        fit = None
    else: 
        aligned_traces = alignments.apply_log(logB, net, initial_marking, final_marking)
        x = 0
        for trace in aligned_traces:
            x =  x + trace["fitness"]
        x = int(x)

        aligned_traces_dataframe = alignments.get_diagnostics_dataframe(logB, aligned_traces)
        fit = x/len(logB)
        prec = pm4py.precision_alignments(logB, net, initial_marking, final_marking)
        # prec = 0
    return aligned_traces_dataframe, fit, prec

# %%
def conformance_declare (model, logB):
    d4py = Declare4Py()
    d4py.parse_xes_log(logB)
    d4py.parse_decl_model(model+".decl")
    model_check_res = d4py.conformance_checking(consider_vacuity=False)
    tot_fitness = []

    for trace, results in model_check_res.items():
        sat = 0
        for constr in results.items():
            if(constr[1].state.value == "Satisfied"): sat+=1
        
    tot_fitness.append(sat/len(results.items()))    
    return mean(tot_fitness)

def conformance_powl (model, logB):
    return