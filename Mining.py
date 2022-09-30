# %%
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import pm4py
import uuid
import os


# %% [markdown]
# ## DISCOVERY INDUCTIVE AND HEURISTIC

# %%
def discovery_inductive(log, uuidstr):
    ind = pm4py.discover_petri_net_inductive(log)
    gviz = pn_visualizer.apply(ind[0], ind[1], ind[2])
    PATH = 'Models'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    filename = "Models/ind_"+uuidstr
    pn_visualizer.save(gviz, filename+".png")
    pm4py.write_pnml(ind[0], ind[1], ind[2], filename+".pnml")
    return ind, filename
def discovery_heuristic(log, uuidstr):
    PATH = 'Models'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    heu = pm4py.discover_petri_net_heuristics(log)
    gviz = pn_visualizer.apply(heu[0], heu[1], heu[2])
    filename = "Models/heu_"+uuidstr
    pn_visualizer.save(gviz, filename+".png")
    pm4py.write_pnml(heu[0], heu[1], heu[2], filename + ".pnml")
    return heu, filename


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

    aligned_traces = alignments.apply_log(logB, net, initial_marking, final_marking)
    x = 0
    for trace in aligned_traces:
        x =  x + trace["fitness"]
    x = int(x)

    aligned_traces_dataframe = alignments.get_diagnostics_dataframe(logB, aligned_traces)
    return aligned_traces_dataframe, x/len(logB)


