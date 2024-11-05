import numpy as np

from humla.evaluate.evaluation_online import compute_online_PF



class Collector:
    def __init__(self,file_source:str):
        self.y_true=[]
        self.y_predict=[]
        self.file_source=file_source
        self.initialize()
        self.PF_results=dict()
        self.Performance_evaluate()

    def Performance_evaluate(self):
        pfs_dct = compute_online_PF(self.y_true, self.y_predict, 0.99)
        self.pfs_dic=pfs_dct
        recall0 = pfs_dct["recall0_tt"]
        recall1 = pfs_dct["recall1_tt"]
        Gmean = pfs_dct["gmean_tt"]
        mcc = pfs_dct["mcc_tt"]
        precision = pfs_dct["precision_tt"]
        f1_score = pfs_dct["f1_score_tt"]
        avg_acc = pfs_dct["ave_acc_tt"]
        self.PF_results["avg_R0"]=np.nanmean(recall0)
        self.PF_results["avg_R1"]=np.nanmean(recall1)
        self.PF_results["avg_Gmean"]=np.nanmean(Gmean)
        self.PF_results["avg_mcc"]=np.nanmean(mcc)
        self.PF_results["avg_precision"]=np.nanmean(precision)
        self.PF_results["avg_f1_score"]=np.nanmean(f1_score)
        self.PF_results["avg_acc"]=np.nanmean(avg_acc)

    def initialize(self):
        with open(self.file_source,"r") as f:
            f.readline()
            line=f.readline()
            while(line):
                datas=line.split()
                y_true=int(datas[1])
                self.y_true.append(y_true)
                y_predict=int(datas[2])
                self.y_predict.append(y_predict)
                line = f.readline()
