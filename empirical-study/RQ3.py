import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats.stats import kendalltau
from metrics.a12.a12 import a12 as A12
import matplotlib.pyplot as plt


system_path=os.path.dirname(__file__)
save_path=os.path.join(system_path,os.path.join('results','RQ2'))
performance_path=os.path.join(save_path, "performance")
performance_path_new=os.path.join(performance_path, "new")
performance_path_old=os.path.join(performance_path, "old")
from data_stream.real_data_stream import data_id_2name

out_put_dir=os.path.join('results','RQ3')

plot_dir=os.path.join("results","rslt.plot")

p_2=[data_id_2name(i) for i in range(1,23,1)]
p_2.append("cross_project")
model_list=["odasc","oob","pbsa","humla-effort-auto","humla-effort-1"]
metrics = ["avg_R0", "avg_R1", "avg_Gmean", "avg_mcc", "avg_precision", "avg_f1_score", "avg_acc"]


def run_rank(metric="avg_Gmean",type="new"):
     """
     calculate the rank for each model in datasets {type}
     :param metric: the basic metric u choose
     :param type: the datasets u choose
     :return:
     """
     out_put_data=[]
     if type=="new":
         performance_dir=performance_path_new
     elif type=="old":
         performance_dir=performance_path_old
     else:
         print("--Wrong type, please choose 'new' or 'old' ")
         return
     for i in range(1,23,1):
         dataset=data_id_2name(i)
         metrc_value=[]
         for model in model_list:
             data=pd.read_csv(os.path.join(os.path.join(performance_dir,dataset),f"{model}.csv"))
             metrc_value.append(data[metric].values[-1])
         rank=np.argsort(metrc_value)+1
         out_put_data.append(rank)
     out_put_data=np.array(out_put_data)
     avg=np.mean(out_put_data,axis=0)
     out_put_data=np.vstack([out_put_data,avg])
     pd.DataFrame(data=out_put_data,index=p_2,columns=model_list).to_csv(os.path.join(out_put_dir,f"rank_{metric}_{type}.csv"))

def Kendall_tau(metric="avg_Gmean"):
    """
    this part will calculate the kendall tau score
    :param metric:
    :return:
    """
    new_data=pd.read_csv(os.path.join(out_put_dir,f"rank_{metric}_new.csv")).values[:,1:]
    old_data=pd.read_csv(os.path.join(out_put_dir,f"rank_{metric}_old.csv")).values[:,1:]
    out_put=[]
    for i in range(23):
        line=[]
        line.append(kendalltau(new_data[i],old_data[i]).correlation)
        # line.append(wilcoxon(new_data[i],old_data[i]).pvalue)
        line.append(A12(new_data[i],old_data[i]))
        out_put.append(line)
    pd.DataFrame(data=out_put,index=p_2,columns=["kendall","A12"]).to_csv(os.path.join(out_put_dir,f"rank_{metric}_statistic.csv"))

def plot_kendall():
    """
    this function will plot the kendall tau socre
    :return:
    """
    value_name = "kendall"
    plt.figure()
    table=[]
    positions=[1,2,3,4,5,6,7]
    for metric in metrics:
        data=pd.read_csv(os.path.join(out_put_dir,f"rank_{metric}_statistic.csv"))[value_name].values
        table.append(data)
    plt.boxplot(table,positions=positions)
    for i,data in enumerate(table):
        plt.scatter(np.full(data.shape,positions[i]),data)
    plt.xticks(positions,["R0", "R1", "G-mean", "MCC", "Precision", "F1- score", "ACC"])
    plt.savefig(os.path.join(plot_dir,'RQ3_Kendall.png'),dpi=350)
    plt.savefig(os.path.join(plot_dir,'RQ3_Kendall.pdf'),bbox_inches='tight')

    # plt.show()

def plot_Rank(metric="avg_Gmean",type="new"):
    """
    this function will plot the rank score for the metric
    :param metric: the basic metric u choose
    :param type: the datasets u choose (new or old)
    :return:
    """
    plt.figure()
    plt.ylim((0, 5))

    colors = plt.cm.Paired(range(5))  # choose the color
    models = ["ODaSC", "OOB", "PBSA", "Eco-HumLa", "HumLa"]
    data = pd.read_csv(os.path.join(out_put_dir, f"rank_{metric}_{type}.csv")).values[-1][1:]

    #
    sorted_indices = sorted(range(len(data)), key=lambda k: data[k])
    sorted_scores = [data[i] for i in sorted_indices]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    #
    bars = plt.bar(range(len(sorted_scores)), sorted_scores, color=sorted_colors, edgecolor='black', linewidth=1,
                   alpha=0.8,tick_label=sorted_models)

    #
    plt.rcParams['font.size'] = 12
    for bar, score in zip(bars, sorted_scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, yval + 0.005, f'{score:.2f}', ha='center', va='bottom')

    #
    plt.title(f'Rank Scores Based on {metric} in {type} Datasets', fontweight='bold')
    plt.ylabel('Rank Score', fontweight='bold')
    plt.tight_layout()  #
    plt.savefig(os.path.join(plot_dir, f'Rank_Score_{metric}_{type}.png'), dpi=350)
    plt.savefig(os.path.join(plot_dir, f'Rank_Score_{metric}_{type}.pdf'), bbox_inches='tight')
    plt.show()


if __name__=="__main__":
    #firstly, run this part to get the rank score for each model
    run_rank("avg_Gmean","new")
    run_rank("avg_Gmean","old")
    run_rank("avg_mcc","new")
    run_rank("avg_mcc","old")
    run_rank("avg_R0", "new")
    run_rank("avg_R0", "old")
    run_rank("avg_R1", "new")
    run_rank("avg_R1", "old")
    run_rank("avg_precision", "new")
    run_rank("avg_precision", "old")
    run_rank("avg_f1_score", "new")
    run_rank("avg_f1_score", "old")
    run_rank("avg_acc", "new")
    run_rank("avg_acc", "old")
    #next, we can know the kendall tau score by the rank score.
    Kendall_tau("avg_Gmean")
    Kendall_tau("avg_mcc")
    Kendall_tau("avg_R0")
    Kendall_tau("avg_R1")
    Kendall_tau("avg_precision")
    Kendall_tau("avg_f1_score")
    Kendall_tau("avg_acc")
    #plot the figure
    plot_kendall()
    plot_Rank("avg_mcc",type="new")
    plot_Rank("avg_mcc",type="old")
    plot_Rank("avg_Gmean", type="new")
    plot_Rank("avg_Gmean", type="old")




