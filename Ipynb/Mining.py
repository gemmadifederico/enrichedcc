#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import pm4py
import uuid
import os


# ## DISCOVERY INDUCTIVE AND HEURISTIC

# In[5]:


def discovery_inductive(log):
    ind = pm4py.discover_petri_net_inductive(log)
    gviz = pn_visualizer.apply(ind[0], ind[1], ind[2])
    PATH = 'Models'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    pn_visualizer.save(gviz, "Models/ind_"+str(uuid.uuid4())+".png")
    return ind
def discovery_heuristic(log):
    PATH = 'Models'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    heu = pm4py.discover_petri_net_heuristics(log)
    gviz = pn_visualizer.apply(heu[0], heu[1], heu[2])
    pn_visualizer.save(gviz, "Models/heu_"+str(uuid.uuid4())+".png")
    return heu


# ## CONFORMANCE - ALIGNMENT

# In[ ]:


def conformance (logA, logB, alg):
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

    aligned_traces_dataframe = alignments.get_diagnostics_dataframe(logA, aligned_traces)
    return aligned_traces_dataframe, x/len(logB)

