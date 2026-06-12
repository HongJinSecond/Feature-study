import os

from pandas.io.sas.sas_constants import dataset_length

from data_stream.real_data_stream import data_id_2name
from preExperiment.RQ1 import *
from metrics.RQ1 import *
import pandas as pd
from metrics.a12.a12 import a12 as a_12
system_path=os.path.dirname(__file__)

save_path=os.path.join(system_path,os.path.join('results','RQ1'))

figure_save_path=os.path.join(save_path,"figure")
table_save_path=os.path.join(save_path,"table")

data_source_path=os.path.join(system_path, "data/data.inuse")
new_data_path=os.path.join(data_source_path,"new")
old_data_path=os.path.join(data_source_path,"old")

#the 14 features we will test
feature_list=['fix','ns','nd','nf','entrophy','la','ld','lt','ndev','age','nuc','exp','rexp','sexp']


#the 5 features that will be impacted
feature_impacted=['lt','ndev','age','nuc','rexp']

projects=[data_id_2name(i) for i in range(1,23,1)]


def analyse_all_projects_for_RQ11(projects:list,features:str,methods:list,out_put_path:str,func:"metric"):
    """
    this function will collect the basic information and result a fix table in RQ1/table/RQ11
    :param projects:
    :param features:
    :param methods:
    :param out_put_path:
    :param func:
    :return:
    """
    methods_name = [method.__name__ for method in methods]
    for feature in features:
        print(f"RQ1.1---------run for featur {feature}----------------")
        data=[]
        for project in projects:
            data_old = pd.read_csv(os.path.join(old_data_path, project + "_vld_st.csv"))
            data_new= pd.read_csv(os.path.join(new_data_path, project+ "_vld_st.csv"))
            line=[]
            for method in methods:
                line.append(method(data_new[feature].values,data_old[feature].values))
            data.append(line)
        pd.DataFrame(data,index=projects,columns=methods_name).to_csv(os.path.join(out_put_path,f"{feature}_{func}.csv"))
#
# def analyse_all_projects_for_RQ12(projects:list,features:str,methods:list,out_put_path:str):
#     for method in methods:
#         print(f"RQ1.2-----------run metric {method.__name__}-------------------")
#         data = []
#         for project in projects:
#             data_old = pd.read_csv(os.path.join(old_data_path, project + "_vld_st.csv"))
#             data_new= pd.read_csv(os.path.join(new_data_path, project+ "_vld_st.csv"))
#             line=[]
#             for feature in features:
#                 line.append(method(data_new[feature].values,data_old[feature].values))
#             data.append(line)
#         pd.DataFrame(data,index=projects,columns=features).to_csv(os.path.join(out_put_path,f"{method.__name__}.csv"))



if __name__ == "__main__":
    #Run the basic table for RQ1, this operation will count all the basic metrics we need, and save it in RQ1/table/results
    for i in range(1,23,1):
        dataset=data_id_2name(i)
        print(f"--------Run RQ1 for project {dataset}-------------")
        data_0=pd.read_csv(os.path.join(old_data_path,dataset+"_vld_st.csv"))
        data_1=pd.read_csv(os.path.join(new_data_path,dataset+"_vld_st.csv"))

        # the results save data path for RQ1.1 1.2 1.3
        basic_results_path=os.path.join(table_save_path,"results")

        # calculate_by_step(data_0,data_1,test_features,[Pearson_correlation,Spearman_correlation],test_size,figure_save_path)
        result_dataFrame=Result_Analyse(data_0,data_1,feature_list,[Percentage,Zero_Norm,L_1_Norm,L_2_Norm,Spearman_correlation,Pearson_correlation])
        result_dataFrame.to_csv(os.path.join(basic_results_path,f"{dataset}_result.csv"))
    RQ11_path=os.path.join(table_save_path,"RQ11" )

    print("-------------Run RQ1.1---------------")
    print("-------------Run RQ1.1 for metrics-------------")
    analyse_all_projects_for_RQ11(projects,feature_impacted,[Percentage,Zero_Norm,L_1_Norm,L_2_Norm,Spearman_correlation,Pearson_correlation],RQ11_path,func="metrics")
    print("------------Run RQ1.1 for statistics")
    analyse_all_projects_for_RQ11(projects,feature_impacted,[Wilcoxon,a_12],RQ11_path,func="statistics")
