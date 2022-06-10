
# Enriched Conformance Checking

The project implements an enriched conformance checking for an hybrid process model capable of representing human behavior. 



## Features

The project is composed by three main python scripts:
- Mining: executes discovery and conformance of imperative process models.
- Statsdata: executes discovery and conformance of the statistics.
- Econformance: orchestrates all the components.
And by two jar files:
- dcr-discovery: discovery of DCR graphs (see https://github.com/tslaats/DisCoveR). 
- dcr-conformance: conformance of DCR graphs.




## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages.

```bash
  pip install numpy
  pip install pandas
  pip install scipy
  pip install matplotlib
  pip install pm4py
```
    
## Usage/Examples

To run the approach execute:

```python
python Econformance.py arg0 arg1 arg2 .. argX
```
- arg0 is the path of the event log used to discover the reference hybrid model
- arg1, arg2, .. , argX is the list of event logs used for the conformance

For example:
```python
python Econformance.py Example/logNormal.xes Example/logAbsence.xes Example/logDelay.xes
```
All the models derived by the application of the process discovery algorithms are saved in the folder /Models.
The fitness values obtained by the application of the confromance checking are saved in the Results.csv file.

