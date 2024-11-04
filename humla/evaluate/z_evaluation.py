"""
This script contains classes that are of higher level than v_evaluation_offline.py or z_evaluation_online.py
"""
from evaluate.evaluation_offline import eval_clf_offline
from evaluate.evaluation_online import eval_clf_online
import numpy as np


class record_pfs:
    """For easy pf record"""

    def __init__(self, n_seed):
        self.n_seed = n_seed
        self.off_pf_names = None
        self.off_pfs = None
        self.on_pf_names = None
        self.on_pfs_aveT = None
        self.on_pfs_tt_acc = None

    def pfs_update_np(self, ss, result_np, theta_pf=0.99):
        off_pfs_, off_metric_names = eval_clf_offline(result_np)
        self.off_pf_names = off_metric_names

        on_pfs_tt_, on_pfs_aveT_, on_metric_names = eval_clf_online(result_np, theta_pf)
        self.on_pf_names = on_metric_names

        if ss == 0:  # init
            self.off_pfs = np.empty((len(off_metric_names), self.n_seed))
            self.on_pfs_aveT = np.empty((len(off_metric_names), self.n_seed))
            self.on_pfs_tt_acc = on_pfs_tt_
        else:  # update on_pfs_tt_acc
            self.on_pfs_tt_acc += on_pfs_tt_
        # update
        self.off_pfs[:, ss] = off_pfs_
        self.on_pfs_aveT[:, ss] = on_pfs_aveT_

    def pfs_update_result(self, ss, clf_result, theta=0.99):
        """
        This method is not really in use as the result.format to deal with is usually numpy
        so that we can use np.savez() to save and load the evaluate.
        Nevertheless, we retain this method in case we need to use it later.

        :param ss: seed place in the return matrix
        :param clf_result: when the format is result.result.result_sdp_model.
        :param theta: for the classifier
        :return:
        """
        off_pfs_, off_metric_names = eval_clf_offline(clf_result)
        self.off_pf_names = off_metric_names

        on_pfs_tt_, on_pfs_aveT_, on_metric_names = eval_clf_online(clf_result, theta)
        self.on_pf_names = on_metric_names

        if ss == 0:  # init
            self.off_pfs = np.empty((len(off_metric_names), self.n_seed))
            self.on_pfs_aveT = np.empty((len(off_metric_names), self.n_seed))
            self.on_pfs_tt_acc = on_pfs_tt_
        else:  # update on_pfs_tt_acc
            self.on_pfs_tt_acc += on_pfs_tt_
        # pfs update
        self.off_pfs[:, ss] = off_pfs_
        self.on_pfs_aveT[:, ss] = on_pfs_aveT_

    def compute_ave_pfs(self):
        """compute average pf-s across seeds
        NOTE we can only call this method after accomplish "pfs_update" of all seeds
        """
        ave_off_pfs = [np.nanmean(self.off_pfs[pp, :]) for pp in range(len(self.off_pf_names))]
        ave_on_pfs = [np.nanmean(self.on_pfs_aveT[pp, :]) for pp in range(len(self.on_pf_names))]
        return ave_off_pfs, ave_on_pfs

    def compute_ave_on_pfs_tt(self):
        """compute average pf-s across seeds at each time steps"""
        return self.on_pfs_tt_acc/self.n_seed
