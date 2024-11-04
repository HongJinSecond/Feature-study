# How to run this project
Writen by Zuowei Chen 2024-10-30

e-mail: 2021113561@stu.hit.edu.cn

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



## Dependency

see requirements.txt

## Epilogue

Have a good time to use our code!

