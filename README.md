# Introduction of this package

This package has two projects. They are "Commit-Guru-Improved" and "Empirical study".

Do not use this directory as the work path. Please change in each project directory and run each project Separately.


# Commit Guru Improved

This project is our proposed improved tool to extract the datasets of a software project. 

**You can check the detail information in [./Commit-Guru-Improved/README.md](./Commit-Guru-Improved/README.md).**

Ingests and analyzes code repositories.

==> This tool is improved based on tool Commit Guru, which improved the feature extraction part.



## Installation

1. Clone this repository in to an empty directory
2. Check and modify `./config.json`, especially the database config.

Db: information relating to your postgresql database setup
logging: information about how to write logging information
gmail: gmail account to be used to send cas notifications
repoUpdates: how often repositories should be updated for new commits
system: how many worker threads the cas system can use to analyze and ingest repos.

### Dependencies

make share you are running in **Linux**
Additional Instructions are available in SETUP.md

* Python  >= 3.3 and <= 3.6
* Pip for Python Version > 3.3 and < 3.6
* Git > 1.7
* R
* python-dev
* rpy2
* requests
* dateutil
* sqlalchemy
* py-postgresql
* GNU grep
* MonthDelta


### Installing rpy2

```
pip install rpy2
```

### Additional Pip Packages

Install the following packages by doing `pip install `  and then the package
name. Make sure you are using python3, such as using a virtualenv if using Ubuntu.

* SQL Alchemy (sqlalchemy)
* psycopg2
* requests (requests)
* python-dateutil (python-dateutil)
* any package the runtime pip prompts you that are missing.

To install the MonthDelta package, simply do: `pip install https://pypi.python.org/packages/source/M/MonthDelta/MonthDelta-1.0b.tar.bz2`

### Attetion

make sure that this project need the dataset tool PostgreSQL

### First-Time Database Setup

Set up the database for the first time by running `python script.py initDb`

## Usage

In a terminal, type `nohup python script.py & ' to start the code repo analyzer and run it in the background.

## OFFLine

Update by Zuowei Chen 2024-10-28
I have found if we once use the script.py to extract datasets, everytime it will use **git clone** to clone the project. However, the clone process may be interrupted due to the network problem and we need restart this tool again, which is very hassle and waste of time, so I have integrates a offline version of this tool, the usage can be seen below:

### Base

```
python script.py initDb
python create_repositories.py {project_name} {url} #In offline version, {url} can be any string since we do not use it.
python offline.py {project_name}#make sure that we have already clone the project to ./CASRepos/git/
```

### Mode

```
python offline.py {project_name} {Mode} #Mode can be chosen from "New" and "Old". "New" is our method and "Old" is the original Commit Guru (Chronological order). You can not input param 'Mode' and it will run default mode "Old". 
```

## GUI

This feature was added in version 2.0. This is an extension of the offline mode.

```
python runGUI.py
```

### GUI display

![](/Commit-Guru-Improved/pic/gui.png)

* Project Name: The target project, make sure you have cloned it to `CASRepos/git/`.
* Mode: `NEW` means BDFE and `Old` means simple chronological order.
* start: If you have prepared the params above, click it and start.
* Output path: The data will be extracted to this path.

### Success

![](/Commit-Guru-Improved/pic/success.png)

### Error

![](/Commit-Guru-Improved/pic/error.png)


# empirical study

This project is our experiment project to answer our RQs about the feature issue. 

**Please check [./empirical-study/README.md](./empirical-study/README.md).**

## Structure

### ./results

This folder contains all the results of the project's run. 

**./RQ1** contains the output for our RQ1. You can check the detail information in RQ1.py

**./RQ2** contains the output for our RQ2. You can check the detail information in RQ2.py.

***./RQ2/output*** need the original output data of each project. For example, if you run the project in new datasets, 
the result will first be saved in **./results/rslt.save** and copy them to **./RQ2/output/new_datastest**; and the same as old.

**./RQ3** contains the output for our RQ3. You can check the detail information in RQ3.py.

**./rslt.plot** contains the figures we generate.

**./rslt.save** contains the outputs of models on different datasets. You can find the full ouput in ./results/RQ2/puput

### ./baselineModel

This folder contains the 3 baseline models. They are OOB, PBSA, OOC.

### ./humla

This folder contains the package of model HumLa and ODaSC

### ./data

This folder contains all the datasets we use.

**./data_sets** contains the original datasets which do not have been pre-processed.

**./data.inuse** contains the datasets we finally used in our experiment.

**new** means the datasets extracted by our new tool

**old** means the datasets extracted by the old _Commit Guru._

**manul**  is the experimental data that we conduct manual checking tests.

### ./data_stream

This folder contains the function which transform the datasets into stream data for online learing models.

## Usage

**main.py** contains the basic function to run the JIT-SDP models.

**RQ1.py** contains the function we use to answer our RQ1.

**RQ2.py** contains the function we use to answer our RQ2.

**RQ3.py** contains the function we use to answer our RQ3.

**preprocess.py** contains the function we use to pre-process the datasets.

### My Experiment

<ol>
<li>Run RQ1.py</li>
<li>Run main.py for the models you want to check</li>
<li>Move the results of models to ./results/RQ2/output/new(old)_datasets</li>
<li>Run RQ2.py</li>
<li>Run RQ3.py</li>
<li>Check Result</li>
</ol>


### Addtion

pvalue_adjustment.ipynb: This is an additional experiment in reference to the reviewers' comments. The results can be checked in **results/RQ2/performance/adjustment**

## Dependency

see requirements.txt

PYTHON=3.6

pip install -r requirements.txt

## Epilogue

Enjoy yourself!



# Supplemental.pdf

This is the supplemental materials for our experimental results.

