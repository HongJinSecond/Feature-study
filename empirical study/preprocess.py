import os

from humla.humla import print_pf
from preExperiment.RQ1 import *
from metrics.RQ1 import *
import pandas as pd
from data_stream.real_data_stream import data_id_2name
from utility import cvt_day2timestamp,cvt_timestamp2day

system_path=os.path.dirname(__file__)
save_path=os.path.join(system_path,'results')
figure_save_path=os.path.join(save_path,"figure")

data_source_path=os.path.join(system_path, "data/data_sets")
new_data_path=os.path.join(data_source_path,"new")
old_data_path=os.path.join(data_source_path,"old")

data_issue_path=os.path.join(system_path, "data/data.inuse")
new_issue_path=os.path.join(data_issue_path,"new")
old_issue_path=os.path.join(data_issue_path,"old")


#the features u will maintain to train your model
keep_dim=['author_date_unix_timestamp','fix','ns','nd','nf','entrophy','la','ld','lt','ndev','age','nuc','exp','rexp','sexp','contains_bug','days_to_first_fix']


def time_stamp_preprocess(data:pd.DataFrame):
    '''
    delete the recent six-months data from the table, if u do not need to preprocess two datasets,
    just change the return only contains new_data[0:i] and input old_data as None.
    :param data:
    :return: the DataFrame which just do not concern
     the recent six months.
    '''
    times = data["author_date_unix_timestamp"].values
    time = times[-1]
    delay = cvt_day2timestamp(180)  # 删除半年的时间
    most_recent_time = time - delay

    time_target = len(data)
    flag=time_target
    for i in range(time_target - 1, -1, -1):
        if times[i] <= most_recent_time:
            flag = i
            return data[0:flag]

def invalid_data_preprocess(data:pd.DataFrame):
    '''
    drop the invalid data from the table
    :param data: the DataFrame that u want to preprocess
    :return: None
    '''

    #first we will delete the Merge commit due to its` 0 value
    data.drop(data[data['classification'] == 'Merge'].index, inplace=True)

    #days-to-first-fix must > 0
    data.drop(data[data['days_to_first_fix']<0].index, inplace=True)

def calculate_days_to_first_fix(data):
    if data['unix_timestamp_first_fix'] == 0.:
        return 0
    else:
        time_stamp=data['unix_timestamp_first_fix']-data['author_date_unix_timestamp']
        return cvt_timestamp2day(time_stamp)

if __name__=="__main__":
    for i in range(1,22,1):
        name=data_id_2name(i)
        print(f"process project ------{name}------")
        input_file_name=name+".csv"
        # the data set must be ordered by it`s commit time-stamp
        new_data=pd.read_csv(os.path.join(new_data_path,input_file_name)).sort_values(['author_date_unix_timestamp','commit_hash'],ascending=[True,True])
        old_data=pd.read_csv(os.path.join(old_data_path,input_file_name)).sort_values(['author_date_unix_timestamp','commit_hash'],ascending=[True,True])
        new_data.to_csv(os.path.join(new_data_path,input_file_name),index=False)
        old_data.to_csv(os.path.join(old_data_path,input_file_name),index=False)

        new_data=pd.read_csv(os.path.join(new_data_path,input_file_name))
        old_data=pd.read_csv(os.path.join(old_data_path,input_file_name))
        #使用新数据集的标签 Only run when u use the dataset at the first time
        for i in range(len(new_data)):
            assert old_data.at[i, 'commit_hash'] == new_data.at[i, 'commit_hash'],f"{input_file_name} 序号不匹配"
            old_data.at[i, 'contains_bug'] = new_data.at[i, 'contains_bug']
            old_data.at[i, 'unix_timestamp_first_fix'] = new_data.at[i, 'unix_timestamp_first_fix']

        new_data.to_csv(os.path.join(new_data_path,input_file_name),index=False)
        old_data.to_csv(os.path.join(old_data_path,input_file_name),index=False)


        #delete the nearlist half-year data
        new_data=time_stamp_preprocess(new_data)
        old_data=time_stamp_preprocess(old_data)
        #
        #可以在此部分加入一些其他预处理操作


        #进行数据的转换
        new_data['fix']=new_data['fix'].astype(int)
        new_data['contains_bug']=new_data['contains_bug'].astype(int)
        new_data['unix_timestamp_first_fix']=new_data['unix_timestamp_first_fix'].fillna(0)
        old_data['fix']=old_data['fix'].astype(int)
        old_data['contains_bug']=old_data['contains_bug'].astype(int)
        old_data['unix_timestamp_first_fix']=old_data['unix_timestamp_first_fix'].fillna(0)

        #calculate the data days-to-first-fix
        old_data['days_to_first_fix']=old_data.apply(calculate_days_to_first_fix,axis=1)
        new_data['days_to_first_fix']=new_data.apply(calculate_days_to_first_fix,axis=1)

        #finally, drop the invalid data
        invalid_data_preprocess(new_data)
        invalid_data_preprocess(old_data)

        out_put_file_name=name+"_vld_st.csv"
        new_data[keep_dim].to_csv(os.path.join(new_issue_path,out_put_file_name),index=False)
        old_data[keep_dim].to_csv(os.path.join(old_issue_path,out_put_file_name),index=False)
