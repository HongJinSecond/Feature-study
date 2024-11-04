from operator import index

import numpy as np
from numpy import newaxis
from pandas.io.pytables import performance_doc
from scipy.stats import wilcoxon
from sqlalchemy.dialects.mssql.information_schema import columns

from metrics.collector import Collector
import pandas as pd
from data_stream.real_data_stream import data_id_2name
from metrics.a12.a12 import a12
import re
import os


system_path=os.path.dirname(__file__)
save_path=os.path.join(system_path,os.path.join('results','RQ2'))

result_output_dir=os.path.join(save_path,"output")

output_source_new=os.path.join(result_output_dir,"new_datasets")
output_source_old=os.path.join(result_output_dir,"old_datasets")


performance_path=os.path.join(save_path, "performance")
performance_path_new=os.path.join(performance_path, "new")
performance_path_old=os.path.join(performance_path, "old")

diff_path=os.path.join(performance_path,"diff")
projects=[data_id_2name(i) for i in range(1,23,1)]

p_2=[data_id_2name(i) for i in range(1,23,1)]
p_2.append("cross_project")

model_list=["odasc","oob","pbsa",os.path.join("our-human","effort-auto"),os.path.join("our-human","effort1")]

columns = ["avg_R0", "avg_R1", "avg_Gmean", "avg_mcc", "avg_precision", "avg_f1_score", "avg_acc"]

columns_2=[
    "R0_new","R0_old","R0_diff","R0_p_value","R0_h0","R0_a12"
    , "R1_new","R1_old","R1_diff","R1_p_value","R1_h0","R1_a12"
    ,"Gmean_new","Gmean_old","Gmean_diff","Gmean_p_value","Gmean_h0","Gmean_a12"
    ,"mcc_new","mcc_old","mcc_diff","mcc_p_value","mcc_h0","mcc_a12"
    ,"precision_new","precision_old","precision_diff","precision_p_value","precision_h0","precision_a12"
    ,"f1_score_new","f1_score_old","f1_score_diff","f1_score_p_value","f1_score_h0","f1_score_a12"
    ,"acc_new","acc_old","acc_diff","acc_p_value","acc_h0","acc_a12"
]

target_str="rslt_test.s"

def find_files(dir:str,target_str):
    file_list=[]

    for root,dirs,files in os.walk(dir):
        for file in files:
            if target_str in file:
                file_list.append(os.path.join(root,file))
    return file_list

def extract_number(path):
    match = re.search(r'rslt_test\.s(\d+)$', path)
    if match:
        return int(match.group(1))
    return float('inf')


def generate_diff():
    """
    Run this function, and it will collect the basic performance for each model and calculate the diff value.
    u can check the result in RQ2/performance/diff
    :return:
    """
    for model in model_list:
        print(f"run for {model}")
        old_data=[]
        new_data=[]
        diff_data=[]
        p_value_data=[]
        a_12_data=[]
        reject_data=[]
        if model == os.path.join("our-human", "effort-auto"):
            model = "humla-effort-auto"
        if model == os.path.join("our-human", "effort1"):
            model = "humla-effort-1"
        for i in range(1, 23, 1):
            p_value_line=[]
            a_12_line=[]
            reject_line=[]
            dataset = data_id_2name(i)

            d_new=pd.read_csv(os.path.join(os.path.join(performance_path_new,dataset), f"{model}.csv"))
            d_old=pd.read_csv(os.path.join(os.path.join(performance_path_old,dataset), f"{model}.csv"))

            new_line=d_new.iloc[30].values
            old_line=d_old.iloc[30].values
            diff_line=np.around((new_line[1:]-old_line[1:]).astype('float'),4)
            old_data.append(old_line[1:])
            new_data.append(new_line[1:])
            diff_data.append(diff_line)
            for m in columns:
                m_new=d_new[m].values[0:30]
                m_old=d_old[m].values[0:30]
                p_value=wilcoxon(m_new,m_old).pvalue
                if p_value<=0.05:
                    reject_line.append("yes")
                    a_12 = a12(m_new, m_old)
                else:
                    reject_line.append("no")
                    a_12 = "nan"
                p_value_line.append(p_value)
                a_12_line.append(a_12)
            p_value_data.append(p_value_line)
            a_12_data.append(a_12_line)
            reject_data.append(reject_line)


        new_csv=pd.DataFrame(old_data,index=projects,columns=columns)
        new_csv.to_csv(os.path.join(diff_path,f"{model}_old.csv"))
        old_csv=pd.DataFrame(new_data,index=projects,columns=columns)
        old_csv.to_csv(os.path.join(diff_path,f"{model}_new.csv"))
        pd.DataFrame(diff_data,index=projects,columns=columns).to_csv(os.path.join(diff_path,f"{model}_diff.csv"))
        p_value_across=[]
        a_12_across=[]
        reject_across=[]
        for m in columns:
            m_new=new_csv[m].values
            m_old=old_csv[m].values
            p_value = wilcoxon(m_new, m_old).pvalue
            if p_value <= 0.05:
                reject_across.append("yes")
                a_12 = a12(m_new, m_old)
            else:
                reject_across.append("no")
                a_12 = "nan"
            p_value_across.append(p_value)
            a_12_across.append(a_12)
        p_value_data.append(p_value_across)
        a_12_data.append(a_12_across)
        reject_data.append(reject_across)
        pd.DataFrame(p_value_data, index=p_2, columns=columns).to_csv(os.path.join(diff_path, f"{model}_p_value.csv"))
        pd.DataFrame(a_12_data, index=p_2, columns=columns).to_csv(os.path.join(diff_path, f"{model}_a_12.csv"))
        pd.DataFrame(reject_data, index=p_2, columns=columns).to_csv(os.path.join(diff_path, f"{model}_reject.csv"))

def count_all_table():
    """
    this function will unit all the values for one model in one table, u can run it if u need.
    :return:
    """
    for model in model_list:
        out=pd.DataFrame(index=range(23))
        if model == os.path.join("our-human", "effort-auto"):
            model = "humla-effort-auto"
        if model == os.path.join("our-human", "effort1"):
            model = "humla-effort-1"
        new_data_table=pd.read_csv(os.path.join(diff_path,f"{model}_new.csv"))
        old_data_table = pd.read_csv(os.path.join(diff_path, f"{model}_old.csv"))
        diff_data_table=pd.read_csv(os.path.join(diff_path,f"{model}_diff.csv"))
        p_value_table=pd.read_csv(os.path.join(diff_path, f"{model}_p_value.csv"))
        h0_table=pd.read_csv(os.path.join(diff_path, f"{model}_reject.csv"))
        a_12_table=pd.read_csv(os.path.join(diff_path, f"{model}_a_12.csv"))
        for m in columns:
            out[f"new_{m}"]=new_data_table[m]
            out[f"old_{m}"]=old_data_table[m]
            out[f"diff_{m}"]=diff_data_table[m]
            out[f"p_value_{m}"]=p_value_table[m]
            out[f"h0_{m}"]=h0_table[m]
            out[f"a12_{m}"]=a_12_table[m]
        out.columns=columns_2
        out.to_csv(os.path.join(diff_path, f"{model}.csv"))

if __name__=="__main__":
    """
    this is the basic part of RQ2, we will evaluate the performance of the 5 models in new datasets and old datasets.

    check the dir path results/RQ2/output, u will find there are "new_datasets" and "old_datasets". They are the basic
    output of the five models on the 22 projects. "new_datasets" means they are run on the new datasets w extracted.
    "old_datasets" is the result run on the old datasets. like "new_datasets/brackets/oob" means the model OOB run on the new datasets for brackets.
    
    the operation below will count the basic output and give the metrics, the data will be saved in RQ2/performance/new(old)
    
    """
    # which datasets version do you want to run (new or old), make sure that you have run the choice "old" and "new"
    datasets_choice="old"

    if datasets_choice=="new":
        performance=performance_path_new
        data_source=output_source_new
    else:
        performance=performance_path_old
        data_source=output_source_old

    idx=[i for i in range(30)]
    idx.append("avg")
    os.makedirs(performance_path_new, exist_ok=True)
    os.makedirs(performance_path_old, exist_ok=True)
    for i in range(18,23,1):
        dataset=data_id_2name(i)
        print(f"--------calculate for {dataset}----------")
        #create the output source
        performance_output_dir=os.path.join(performance,dataset)
        os.makedirs(performance_output_dir,exist_ok=True)
        for model in model_list:
            print(f"        ---------for model {model} -------------------")
            #load datasets
            data=[]
            source=os.path.join(data_source,os.path.join(dataset,model))
            files=find_files(source,target_str)
            files=sorted(files, key=extract_number)
            for file in files[0:30]:
                # print(file)
                collector=Collector(file)

                output=[collector.PF_results["avg_R0"],
                    collector.PF_results["avg_R1"],
                    collector.PF_results["avg_Gmean"],
                    collector.PF_results["avg_mcc"],
                    collector.PF_results["avg_precision"],
                    collector.PF_results["avg_f1_score"],
                    collector.PF_results["avg_acc"]]

                data.append(output)
            #the model name contains the character "\\", which will lead a system error. so we decide to change the name when save csv
            if model == os.path.join("our-human","effort-auto"):
                model ="humla-effort-auto"
            if model == os.path.join("our-human","effort1"):
                model ="humla-effort-1"
            #calculate the mean
            data=np.array(data)
            data=np.vstack([data,data.mean(axis=0)[newaxis]])
            pd.DataFrame(data,index=idx,columns=columns).to_csv(os.path.join(performance_output_dir,f"{model}.csv"))
    generate_diff()
    count_all_table()


