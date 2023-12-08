# %%
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.statistics.sojourn_time.log import get as soj_time_get
import statistics as stats
import pm4py
import pandas as pd
from decimal import Decimal
from pm4py.objects.log.importer.xes import importer as xes_importer
from scipy.stats import norm
from pm4py.statistics.attributes.log import get as attributes_get
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# %% [markdown]
# ## Log import
# ### Import of logs for testing

# %%
# logA = xes_importer.apply("../logNormal.xes")
# logB = xes_importer.apply("../logFreq.xes")

# %% [markdown]
# ## Variants

# %%
# Function to get the variants of a log
def get_variants(log):
    variants = case_statistics.get_variant_statistics(log)
    return(variants)

# %% [markdown]
# ## Activity Frequency

# %%
# Function to calculate the the frequency of each event in all the cases
def freq_attributes(log):
    attr_list = pm4py.statistics.attributes.log.get.get_attribute_values(log, attribute_key="concept:name")
    attr_list_freq = dict.fromkeys(attr_list, 0)
    temp = attr_list_freq
    i = 0
    for trace in log:
        temp = dict.fromkeys(temp, 0)
        i = i+1
        for event in trace:
            cn = event.get("concept:name")
            temp[cn] = temp[cn] +1 
        for a, value in attr_list_freq.items():
            if(value == 0): value = []
            value.append(temp[a])
            attr_list_freq[a] = value
    return attr_list_freq

def get_activity_freq_stats(log):
    """
    Get frequency statisics

    Parameters
    --------------
    log
        Log

    Returns
    --------------
    map
        "Sum" : fsum, "Mean": fmean, "Median": fmedian, "StDev": fstdev, "Min": fmin, "Max": fmax
    list
        Attrbute values
    """
    fr = freq_attributes(log)
    fmean = {}
    fmedian = {}
    fmin = {}
    fmax = {}
    fstdev = {}
    fsum = {}
    # Sum
    for key, value in fr.items():
        fsum[key] = sum(value)
    # Mean
    for key, value in fr.items():
        fmean[key] = stats.mean(value)
    # Median
    for key, value in fr.items():
        fmedian[key] = stats.median(value)
    # Min
    for key, value in fr.items():
        fmin[key] = min(value)
    # Max
    for key, value in fr.items():
        fmax[key] = max(value)
    # St Dev
    # If there is only one element, the st dev is 0
    for key, value in fr.items():
        if(len(value) == 1):
            fstdev[key] = 0
        else: fstdev[key] = stats.stdev(value)

    # print(f"Sum: {fsum}, \n Mean {fmean}, \n Median {fmedian}, \n StDev {fstdev}, \n Min {fmin}, \n Max {fmax}")
    attr_list = pm4py.statistics.attributes.log.get.get_attribute_values(log, attribute_key="concept:name")
    return({"Sum" : fsum, "Mean": fmean, "Median": fmedian, "StDev": fstdev, "Min": fmin, "Max": fmax}, dict.fromkeys(attr_list, 0))
    # return(fsum, fmean, fmedian, fstdev, fmin, fmax)

# %% [markdown]
# ## Activity Start/Completion Time

# %%
# Function to calculate the avg start and complete time, and the median start and completion time
def activity_time(log, attr):
    attr_list = pm4py.statistics.attributes.log.get.get_attribute_values(log, attribute_key="concept:name")
    attr_list_time = dict.fromkeys(attr_list, 0)
    for trace in log:
        for event in trace:
            temp = attr_list_time.get(event.get("concept:name"))
            if(temp == 0):
                temp = [] 
            temp.append(event.get(attr))
            attr_list_time[event.get("concept:name")] = temp
    return attr_list_time

def get_activity_start_complete_time(log):
    activity_start_time = activity_time(log, "start_timestamp")
    activity_completion_time = activity_time(log, "time:timestamp")

    x= {}
    y= {}
    j= {}
    k= {}
    for key in activity_start_time.keys():
        mean_start_time = pd.to_timedelta(pd.Series(activity_start_time[key]).dt.hour, unit='H').mean()
        x[key] = mean_start_time
    for key in activity_completion_time.keys():
        mean_completion_time = pd.to_timedelta(pd.Series(activity_completion_time[key]).dt.hour, unit='H').mean()
        y[key] = mean_completion_time
    for key in activity_start_time.keys():
        median_start_time = pd.to_timedelta(pd.Series(activity_start_time[key]).dt.hour, unit='H').median()
        j[key] = median_start_time
    for key in activity_completion_time.keys():
        median_completion_time = pd.to_timedelta(pd.Series(activity_completion_time[key]).dt.hour, unit='H').median()
        k[key] = median_completion_time
    return({"Mean_start_time": x, "Mean_completion_time" : y, "Median_start_time": j, "Median_completion_time" : k})


# %% [markdown]
# ## Activity Duration

# %%
# Function to get the avg duration of each activity identifier in a log
def get_activity_duration(log):
    soj_time = soj_time_get.apply(log, parameters={soj_time_get.Parameters.TIMESTAMP_KEY: "time:timestamp", soj_time_get.Parameters.START_TIMESTAMP_KEY: "start_timestamp"})
    return soj_time

# %%
def get_activity_duration_stats(log, minutes: bool = False):
    if(not "time:timestamp" in pm4py.stats.get_event_attributes(log)):
        #raise ValueError('No timestamp attribute defined...')
        return
    dmean = soj_time_get.apply(log, parameters={soj_time_get.Parameters.TIMESTAMP_KEY: "time:timestamp", soj_time_get.Parameters.START_TIMESTAMP_KEY: "start_timestamp", soj_time_get.Parameters.AGGREGATION_MEASURE: 'mean'})
    dmedian = soj_time_get.apply(log, parameters={soj_time_get.Parameters.TIMESTAMP_KEY: "time:timestamp", soj_time_get.Parameters.START_TIMESTAMP_KEY: "start_timestamp", soj_time_get.Parameters.AGGREGATION_MEASURE: 'median'})
    dmin = soj_time_get.apply(log, parameters={soj_time_get.Parameters.TIMESTAMP_KEY: "time:timestamp", soj_time_get.Parameters.START_TIMESTAMP_KEY: "start_timestamp", soj_time_get.Parameters.AGGREGATION_MEASURE: 'min'})
    dmax = soj_time_get.apply(log, parameters={soj_time_get.Parameters.TIMESTAMP_KEY: "time:timestamp", soj_time_get.Parameters.START_TIMESTAMP_KEY: "start_timestamp", soj_time_get.Parameters.AGGREGATION_MEASURE: 'max'})
    
    
    
    dstdev = get_dur_stdev(log)
    if(minutes == True):
        for act, val in dmean.items():
            mins = val / 60;
            dmean[act] = round(mins)
        for act, val in dmedian.items():
            mins = val / 60;
            dmedian[act] = round(mins)
        for act, val in dmin.items():
            mins = val / 60;
            dmin[act] = round(mins)
        for act, val in dmax.items():
            mins = val / 60;
            dmax[act] = round(mins)
        for act, val in dstdev.items():
            mins = val / 60;
            dstdev[act] = round(mins)
    
    return({"Mean": dmean, "Median": dmedian, "Min": dmin, "Max": dmax, 'StDev': dstdev})

# %%
# Function to get the stdev of the duration of each activity identifier in a log
def get_dur_minmax(log):
    attr_list = pm4py.statistics.attributes.log.get.get_attribute_values(log, attribute_key="concept:name")
    d = {}
    for act in attr_list.keys():
        d[act] = []
    for trace in log:
        for event in trace:
            end = event.get("time:timestamp")
            start = event.get("start_timestamp")
            duration = end - start    
            duration_in_s = round(duration.total_seconds())
            d[event.get("concept:name")].append(duration_in_s)
    minr = {}
    maxr = {}
    for activity, values in d.items():
        minv = min(values)
        maxv = max(values)
        minr[activity] = minv
        maxr[activity] = maxv
    return(minr, maxr)

# %%
# Function to get the stdev of the duration of each activity identifier in a log
def get_dur_stdev(log):
    attr_list = pm4py.statistics.attributes.log.get.get_attribute_values(log, attribute_key="concept:name")
    d = {}
    for act in attr_list.keys():
        d[act] = []
    for trace in log:
        for event in trace:
            end = event.get("time:timestamp")
            start = event.get("start_timestamp")
            duration = end - start    
            duration_in_s = round(duration.total_seconds())
            d[event.get("concept:name")].append(duration_in_s)
    stdev = {}
    for activity, values in d.items():
        if len(values) == 1:
            mm = 0
        else: mm = stats.stdev(values)
        if(mm < 1):
            mm = 0
        stdev[activity] = mm
    return(stdev)

# %%
# Function to get the median of the duration of each activity identifier in a log
def get_dur_median(log):
    attr_list = pm4py.statistics.attributes.log.get.get_attribute_values(log, attribute_key="concept:name")
    d = {}
    for act in attr_list.keys():
        d[act] = []
    for trace in log:
        for event in trace:
            end = event.get("time:timestamp")
            start = event.get("start_timestamp")
            duration = end - start    
            duration_in_s = round(duration.total_seconds())
            d[event.get("concept:name")].append(duration_in_s)
    medians = {}
    for activity, values in d.items():
        mm = stats.median(values)
        medians[activity] = mm
    return(medians)

# %% [markdown]
# ### Activity position

# %%
# Check the index position, and the respective frequency, of each activity in the log
def get_freq_position_normalized_old(log):
    attr_list = pm4py.stats.get_event_attribute_values(log, "concept:name")
    res = {}
    for a in attr_list:
        res[a] = pm4py.stats.get_activity_position_summary(log, activity=a)

    for activity, values in res.items():
        maxres = max(values.values())
        minres = 0
        if(maxres == minres and maxres != 0 and minres != 0):
            for pos, value in values.items():
                res[activity][pos] = 1.0
        else:
            for pos, value in values.items():
                res[activity][pos] = (value - minres)/(maxres - minres)
    return res

# %%
# Check the index position, and the respective frequency, of each activity in the log
def get_freq_position_normalized(log):
    attr_list = pm4py.stats.get_event_attribute_values(log, "concept:name")
    attr_keys = {key: Counter() for key in attr_list}
    properties = pm4py.utils.get_properties(log)
    activity_key = properties[
        pm4py.utils.constants.PARAMETER_CONSTANT_ACTIVITY_KEY] if pm4py.utils.constants.PARAMETER_CONSTANT_ACTIVITY_KEY in properties else pm4py.utils.xes_constants.DEFAULT_NAME_KEY

    for trace in log:
        for i in range(len(trace)):
            lt = len(trace)
            this_act = trace[i][activity_key]
            # Let's compute the percentage position of this event in the trace
            percev = round((100*i)/lt)
            if(percev >= 0 and percev <= 24):
                attr_keys[this_act][1]+=1
            elif(percev >= 25 and percev <= 49):
                attr_keys[this_act][2]+=1
            elif(percev >= 50 and percev <= 74):
                attr_keys[this_act][3]+=1
            elif(percev >= 75 and percev <= 100):
                attr_keys[this_act][4]+=1

    # And then we normalize it
    for act, counts in attr_keys.items():
        maxx = max(counts.values())
        minn = 0
        for k, v in counts.items():
            if(maxx == minn):
                normv = 1
            else:
                normv = (v-minn)/(maxx-minn)
            attr_keys[act][k] = normv

    return attr_keys

# %% [markdown]
# ### Trace length

# %%
def get_trace_length_stats(log):
    temp = []
    #pm4py.objects.log.util.index_attribute.insert_trace_index_as_event_attribute(log)
    for trace in log:
        temp.append(len(trace))
    stdev = 0
    if(len(temp) >1): stdev = stats.stdev(temp)

    return({"Mean": stats.mean(temp), "Median": stats.median(temp), "StDev": stdev, "Min": min(temp), "Max": max(temp)})

# %% [markdown]
# ## STATISTICAL CONFORMANCE

# %% [markdown]
# ### Frequency fitness

# %%
# Fitness of frequencies based on the normal distribution of freq
def get_freq_fitness(u, logBase, logComp):
    result = {}
    i = 0
    attr_list = pm4py.statistics.attributes.log.get.get_attribute_values(logBase, attribute_key="concept:name")

    for trace in logComp:
        temp = dict.fromkeys(attr_list, None)
        i = i+1
        fitness = []
        for event in trace:
            cn = event.get("concept:name")
            if(cn in temp.keys()):
                if(temp[cn] is None): #First initialization
                    temp[cn] = 1
                else:
                    temp[cn] = temp[cn] +1 
        # now that I counted the occurences in this trace, I compare the values with the normal one
        # for each temp activity, I check the probability of the actual activity frequency to fit
        # in the normal distribution of the normal log, using the normal mean and stdev
        for key, value in temp.items():
            if(value is None):
                # I've never observed this activity in the log to compare (but I have it in the reference model as it comes from attr_list)
                # therefore the fitness is 0
                fitness.append(0)
            else: 
                # nAvg = u.get(avgtype).get(key)
                nAvg = u.get('Mean').get(key)
                # nMedian = u.get("Median").get(key)
                nStdev = u.get("StDev").get(key)
                nMin = u.get("Min").get(key)
                nMax = u.get("Max").get(key)
                #  worst case in which the value is < nMin or > nMax
                if value < nMin or value > nMax:
                    fitness.append(0)
                # optimum case where the stdev = 0 and avg = value, meaning that there is a perfect fit
                elif(nStdev == 0 and value == nAvg): 
                    fitness.append(1)
                else:
                    f = norm.pdf(value, loc = nAvg , scale = nStdev)
                    g = norm.pdf(nAvg, loc = nAvg , scale = nStdev)
                    fitness.append(round(f/g,5))
        for k in temp.keys():
            temp[k] = ""
        # trace index: trace.attributes.get("concept:name")
        result[trace.attributes.get("concept:name")] = fitness
    tot = {}
    for index, values in result.items():
        tot[index] = np.mean(values)
    return(result, tot, np.mean(list(tot.values())))

# %%


#########################################
# The assumption is that if stdev < 1, meaning that the difference is lower than 1 seconds, we round it to zero
#########################################

# Fitness of duration based on the normal distribution of the normal log
# and comparing the probability of the new log of being in the distribution function
def get_duration_fitness(u, logBase, logComp):
    result = {}
    attr_list = pm4py.statistics.attributes.log.get.get_attribute_values(logBase, attribute_key="concept:name")

    for trace in logComp:
        ddict = dict.fromkeys(attr_list, None)
        # ddict = {}
        fitness = []
        for event in trace:
            end = event.get("time:timestamp")
            start = event.get("start_timestamp")
            duration = end - start     
            duration_in_s = round(duration.total_seconds())
            # duration_in_s = round(duration.total_seconds()/60)
            cn = event.get("concept:name")
            if (cn in ddict.keys()):
                if(ddict[cn] is None): #First initialization
                    ddict[cn] = [duration_in_s]
                else:
                    ddict[cn].append(duration_in_s)    
            # else: ddict[event.get("concept:name")] = [duration_in_s]
        for activity, values in ddict.items():
            if values is None:
                # I've never observed this activity in the log to compare (but I have it in the reference model as it comes from attr_list)
                # therefore the fitness is 0
                fitness.append(0)
            else:
                dstdev = u.get("StDev").get(activity)
                dmean = u.get("Mean").get(activity)
                ccmean = stats.mean(values)
                if dstdev == 0 and dmean == ccmean: fitness.append(1)
                elif(Decimal(min(values)) < Decimal(u.get("Min").get(activity)) or Decimal(max(values)) > Decimal(u.get("Max").get(activity))):
                    fitness.append(0)
                else:
                    f = norm.pdf(ccmean, loc = dmean , scale = dstdev)
                    g = norm.pdf(dmean, loc = dmean , scale = dstdev)
                    fitness.append(round(f/g,5))
        result[trace.attributes.get("concept:name")] = fitness
    tot = {}
    for index, values in result.items():
        tot[index] = np.mean(values)    
    return(result, tot, np.mean(list(tot.values())))

# %%
def get_evdistribution_intersection(logNormal, logComp, distr_type="hours"):
    """
    Compute the intersection of the distribution of events over time

    Parameters
    ----------------
    logNormal
        Event log
    logComp
        Event log
    distr_type
        Type of distribution (default: days_week):
        - days_month => Gets the distribution of the events among the days of a month (from 1 to 31)
        - months => Gets the distribution of the events among the months (from 1 to 12)
        - years => Gets the distribution of the events among the years of the event log
        - hours => Gets the distribution of the events among the hours of a day (from 0 to 23)
        - days_week => Gets the distribution of the events among the days of a week (from Monday to Sunday)
        - weeks => Gets the distribution of the events among the weeks of a year (from 0 to 52)
    """
    # pm4py.view_events_distribution_graph(logS1d, distr_type="hours", format="png")
    x, y = attributes_get.get_events_distribution(logNormal, distr_type=distr_type, parameters=pm4py.utils.get_properties(logNormal))
    x1, y1 = attributes_get.get_events_distribution(logComp, distr_type=distr_type, parameters=pm4py.utils.get_properties(logComp))

    if(max(y) > max(y1)) : max_hist = max(y) 
    else: max_hist = max(y1)
    if(min(y) > min(y1)) : min_hist = min(y) 
    else: min_hist = min(y1)

    hist_1, _ = np.histogram(y, range=[min_hist,max_hist])
    hist_2, _ = np.histogram(y1, range=[min_hist,max_hist])

    def return_intersection(hist_1, hist_2):
        minima = np.minimum(hist_1, hist_2)
        intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
        return intersection

    intersection = return_intersection(hist_1, hist_2)

    if distr_type == "days_month":
        title = "Distribution of the Events over the Days of a Month";
        x_axis = "Day of month";
    elif distr_type == "months":
        title = "Distribution of the Events over the Months";
        x_axis = "Month";
    elif distr_type == "years":
        title = "Distribution of the Events over the Years";
        x_axis = "Year";
    elif distr_type == "hours":
        title = "Distribution of the Events over the Hours";
        x_axis = "Hour (of day)";
    elif distr_type == "days_week":
        title = "Distribution of the Events over the Days of a Week";
        x_axis = "Day of the Week";
    elif distr_type == "weeks":
        title = "Distribution of the Events over the Weeks of a Year";
        x_axis = "Week of the Year";

    plt.plot(y, 'bo', alpha=0.5)
    plt.plot(y1, 'ro', alpha=0.5)
    plt.xlabel(x_axis)
    plt.ylabel('Number of events')
    plt.title(title)

    plt.figure()
    plt.xlabel(x_axis)
    plt.ylabel('Number of events')
    plt.title(title)
    plt.bar(x,y, color='b', alpha=0.5)
    plt.bar(x1,y1, color='r',  alpha=0.5)
    plt.show()

    return intersection


# %%
# get_evdistribution_intersection(logS1n, logTest3)
# x, y = attributes_get.get_events_distribution(logS1n, distr_type="hours", parameters=pm4py.utils.get_properties(logS1n))
# print(y)

def get_freq_hour_normalized2(log):
    """
    Return the frequency of each activity, per hour, normalized

    Parameters
    ----------------
    log
        Event log
    """
    attr_list = pm4py.statistics.attributes.log.get.get_attribute_values(log, attribute_key="concept:name")
    res = {key: {} for key in attr_list}

    for trace in log:
        for event in trace:
            if len(res[event.get("concept:name")]) == 0:
                # The dict is empty
                res[event.get("concept:name")][event.get("start_timestamp").hour] = 1
            elif event.get("start_timestamp").hour not in res[event.get("concept:name")]:
                res[event.get("concept:name")][event.get("start_timestamp").hour] = 1
            else:
                res[event.get("concept:name")][event.get("start_timestamp").hour] += 1

    for activity, values in res.items():
        # The normalization is between zero and the max value that each activity can have
        maxres = max(values.values())
        # minres = min(values.values())
        minres = 0
        for hour, value in values.items():
            res[activity][hour] = (value - minres)/(maxres - minres)
    return res

def get_time_fitness2(logA, logB):
    nlog = get_freq_hour_normalized2(logA)
    attr_list = nlog.keys()
    trace_fitness = {} 
    for trace in logB:
        x = {key: {} for key in attr_list}
        fitness = []
        for event in trace:
            if len(x[event.get("concept:name")]) == 0:
                # The dict is empty
                x[event.get("concept:name")][event.get("start_timestamp").hour] = 1
            elif event.get("start_timestamp").hour not in x[event.get("concept:name")]:
                x[event.get("concept:name")][event.get("start_timestamp").hour] = 1
            else:
                x[event.get("concept:name")][event.get("start_timestamp").hour] += 1
        for act, values in x.items():
            for hour,freq in values.items():
                # we have to normalize each value:
                freq_norm = (freq - 0)/(max(values.values()) - 0)
                if hour not in nlog[act]:
                    fitness.append(0)
                # The frequency in the normal log is zero, but the activity is executed in the log to compare
                elif freq_norm > 0 and nlog[act][hour] == 0:
                    fitness.append(0)
                elif freq_norm >= nlog[act][hour]:
                    fitness.append(1)
                elif freq_norm < nlog[act][hour]:
                    fitness.append( freq_norm/nlog[act][hour])
        trace_fitness[trace.attributes.get("concept:name")] = fitness
    result = {}
    for id, trace in trace_fitness.items():
        result[id] = np.mean(trace)
    return(trace_fitness, result, np.mean(list(result.values())))

# %%
def get_freq_hour_normalized(log):
    """
    Return the frequency of each activity, per hour, normalized

    Parameters
    ----------------
    log
        Event log
    """
    if(not "time:timestamp" in pm4py.stats.get_event_attributes(log)):
        #raise ValueError('No timestamp attribute defined...')
        return
    # temp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    # res = {key: 0 for key in temp}
    attr_list = pm4py.statistics.attributes.log.get.get_attribute_values(log, attribute_key="concept:name")

    res = {}
    for a in attr_list:
        res[a] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}

    for trace in log:
        for event in trace:
            res[event.get("concept:name")][event.get("start_timestamp").hour] += 1

    for activity, values in res.items():
        maxres = max(values.values())
        minres = min(values.values())
        for hour, value in values.items():
            res[activity][hour] = (value - minres)/(maxres - minres)
    return res

def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

def get_time_fitness(logA, logB):
    nlog = get_freq_hour_normalized(logA)
    nlog2 = get_freq_hour_normalized(logB)
    arr = []
    inters = {}

    for activity, normalized in nlog.items():
        # plt.bar(normalized.keys(),normalized.values(), alpha= 0.5)
        # plt.bar(nlog2[activity].keys(), nlog2[activity].values(), alpha= 0.5)
        # plt.title(activity)
        # plt.show()

        for index, value in normalized.items():
            if activity not in nlog2.keys():
            # if(nlog2[activity][index] is None):
                # In this case, the log to compare does not include the activity stored in the stats, and we return a fitness of 0
                arr.append(0)
            elif(value == 0 and nlog2[activity][index] == 0):
                pass
            elif value == 0: arr.append(0.0)
            elif nlog2[activity][index] >= value:  arr.append(1)
            else:
                arr.append(nlog2[activity][index]/value)

        # hist_1, _ = np.histogram(list(nlog[activity].values()), range=[0,1])
        # hist_2, _ = np.histogram(list(nlog2[activity].values()), range=[0,1])
        # inters[activity] = return_intersection(hist_1, hist_2)
    return(arr, sum(arr)/len(arr))

# %% [markdown]
# ### Position fitness

# %%
# Check the frequency in the given position of each activity in the log
def get_position_fitness_old(nlog, log2):
    logtest = get_freq_position_normalized(log2)
    arr = []
    inters = {}

    for activity, normalized in nlog.items():
        if(not activity in logtest.keys()):
            arr.append(0)
        else:
            for index, value in normalized.items():
                if(not index in logtest[activity].keys()):
                    arr.append(0)
                elif logtest[activity][index] > value:  arr.append(1)
                else:
                    arr.append(logtest[activity][index]/value)
        for key in logtest[activity].keys():
            if not key in normalized:
                arr.append(0)
    return(arr, sum(arr)/len(arr))
    
def get_position_fitness(basecase, testlog):
    attr_keys = {key: Counter() for key in basecase}
    attr_keys2 = {key: Counter() for key in basecase}
    result = {}
    for trace in testlog:
        fitnesst = []
        for i in range(len(trace)):
            this_act = trace[i]["concept:name"]
            lt = len(trace)
            # Here we already know that is zero, because the event does not appear at all in the base case
            if(this_act in basecase.keys()):
                # Let's compute the percentage position of this event in the trace
                percev = (100*i)/lt
                if(percev >= 0 and percev <= 24):
                    attr_keys[this_act][1]+=1
                elif(percev >= 25 and percev <= 49):
                    attr_keys[this_act][2]+=1
                elif(percev >= 50 and percev <= 74):
                    attr_keys[this_act][3]+=1
                elif(percev >= 75 and percev <= 100):
                    attr_keys[this_act][4]+=1
                # And then we normalize it (we normalize because we grouped in chunks)
                for pos, counts in attr_keys[this_act].items():
                    maxx = max(attr_keys[this_act].values())
                    minn = 0
                    if(maxx == minn):
                        normv = 1
                    else:
                        normv = (counts-minn)/(maxx-minn)
                    attr_keys2[this_act][pos] = normv
        for act, count in attr_keys2.items():
            for p, freq in count.items():
                if(not act in basecase.keys()):
                    fitnesst.append(0)
                elif(not str(p) in basecase[act].keys()):
                    fitnesst.append(0)
                elif(basecase[act][str(p)] == freq):
                    fitnesst.append(freq)
                elif(basecase[act][str(p)] > freq):
                    # print("Base case higher")
                    fitnesst.append(1) 
                    # fitnesst.append(freq/basecase[act][p])
                elif(basecase[act][str(p)] < freq):
                    # print("Base case lower")  
                    fitnesst.append(1) 
                    # fitnesst.append(basecase[act][p]/freq)  
                elif(not str(p) in basecase[act].values()):
                    fitnesst.append(0) 
        result[trace.attributes.get("concept:name")] = sum(fitnesst)/len(fitnesst)
    return result, np.mean(list(result.values()))

# %% [markdown]
# ### Compare trace length

# %%
def get_length_fitness(statsl, logTest):
    res = {}
    for trace in logTest:
        value = len(trace)
        #  worst case in which the value is < nMin or > nMax
        if value < statsl.get("Min") or value > statsl.get("Max"):
            fitness = 0
        # optimum case where the stdev = 0 and avg = value, meaning that there is a perfect fit
        elif(statsl.get("StDev") == 0 and value == statsl.get("Mean")): 
            fitness = 1
        else:
            f = norm.pdf(value, loc = statsl.get("Mean") , scale = statsl.get("StDev"))
            g = norm.pdf(statsl.get("Mean"), loc = statsl.get("Mean") , scale = statsl.get("StDev"))
            fitness = round(f/g,5)
        res[trace.attributes.get("concept:name")] = fitness
    return(res, np.mean(list(res.values())))