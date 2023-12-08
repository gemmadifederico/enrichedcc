
# Enriched Conformance Checking

The project implements an enriched conformance checking for an hybrid process model capable of representing human behavior. 



## Features

The project is composed by four main python scripts:
- Mining: executes discovery and conformance of the control flow dimension.
- Statsdata: executes discovery and conformance of the data dimension.
- Discovery: to run the discovery of the multi dimension model.
- Conformance: to run the conformance on the multi dimension model.

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

Install Declare4py, the version used in this project is the following https://github.com/francxx96/declare4py.
    
## Usage/Examples

The approach is now divided into discovery and conformance.
To run the discovery execute: 
```python
python Discovery.py path_xes path_discovered_model
```
- path_xes is the path of the event log used to discover the reference model (without extension)
- path_discovered_model is the path of the folder where to save the discovered models

For example:
```python
python Discovery.py Example/Scenario2/logNormal Example/Scenario2
```
All the models derived by the application of the process discovery algorithms are saved in the folder /Models.


To run the conformance execute:
```python
python Conformance.py path_xes_test path_xes_train path_discovered_model, case, exp_name
```
- path_xes_test is the path of the event log used to for the conformance (without extension)
- path_xes_train is the path of the event log previously used for the discovery
- path_discovered_model is the path of the folder where the discovered models are saved, i.e. the Models folder
- case and exp_name are attributes used to distinguish the different runs in the Results.csv file (case is a int, exp_name is the experiment name and it's a string)

For example:
```python
python Conformance.py Example/Scenario2/logAsence Example/Scenario2/logNormal Example/Scenario2/Models 1 test1
```

The fitness values obtained by the application of the confromance checking are saved in the Results.csv file.

## Badges

[![DOI](https://zenodo.org/badge/501979701.svg)](https://zenodo.org/badge/latestdoi/501979701)
