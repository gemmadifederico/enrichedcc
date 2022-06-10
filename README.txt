To run the approach execute:
python Econformance.py arg0 arg[1..X]

arg0 is the path of the event log used to discover the reference enriched model
arg1, arg2, .. , argX is the list of event logs used for the conformance

E.g.
python Econformance.py Logs/Scenario1/logNormal.xes Logs/Scenario1/logFreq.xes Logs/Scenario1/logDur.xes