import numpy as np
import warnings


# silence the warning
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


def eval_clf_online(result_np, theta_pf=0.99):
    """for pp-report
    :param result_np: (time, y_true, y_pred), created in xxx.jit_sdp_1call()
    :param theta_pf: the para theta_imb to evaluate the online pf
    :return:
    Liyan on 2021-10-12
    """
    # print(info_str) >> the index is 0: time, 1: y_true, 2: y_pred
    actual_labels = result_np[:, 1]
    predict_labels = result_np[:, 2]

    recall0_tt, recall1_tt, gmean_tt = compute_online_PF(actual_labels, predict_labels, theta_pf)
    gmean_tt = gmean_tt.reshape(len(gmean_tt))
    recall0_tt = recall0_tt.reshape(len(recall0_tt))
    recall1_tt = recall1_tt.reshape(len(recall1_tt))
    # pp-report, (3, #steps)
    pf_metrics_tt = np.vstack((gmean_tt, recall0_tt, recall1_tt))

    # ave across time steps
    gmean_ave = np.nanmean(gmean_tt, 0)
    recall0_ave = np.nanmean(recall0_tt, 0)
    recall1_ave = np.nanmean(recall1_tt, 0)
    # pp-report, 1*3
    pf_metrics_ave = (gmean_ave, recall0_ave, recall1_ave)

    metric_names = np.array(("gmean", "recall0", "recall1"))
    return pf_metrics_tt, pf_metrics_ave, metric_names


def Gmean_compute(recall):
    Gmean = 1
    for r in recall:
        Gmean = Gmean * r
    Gmean = pow(Gmean, 1/len(recall))
    return Gmean


def pf_epoch(S, N, theta, t, y_t, p_t):
    if t == 0:
        c = int(y_t)  # class 0 or 1
        S[t, c] = (y_t == p_t)
        N[t, c] = 1
    else:
        S[t, :] = S[t-1, :]
        N[t, :] = N[t-1, :]
        c = int(y_t)  # class 0 or 1
        S[t, c] = (y_t == p_t) + theta * (S[t-1, c])
        N[t, c] = 1 + theta * N[t-1, c]

    recall = S[t, :] / N[t, :]
    gmean = Gmean_compute(recall)
    return recall, gmean


def compute_online_PF(y_tru, y_pre, theta_eval=0.99):
    """
    para theta_eval: used in the online PF evaluation, theta_eval=0.99 by default
    reference: 2013_[JML, #, Leandro based] On evaluate stream learning algorithm

    2021-9      Shuxian helps with creating this method
    2021-12-7   Liyan updates this method slightly making it easier to read
    """
    S = np.zeros([len(y_tru), 2])
    N = np.zeros([len(y_tru), 2])
    recalls_tt = np.zeros([len(y_tru), 2])
    Gmean_tt = np.zeros([len(y_tru), ])
    for t in range(len(y_tru)):
        y_t = y_tru[t]
        p_t = y_pre[t]
        [recalls_tt[t, :], Gmean_tt[t]] = pf_epoch(S, N, theta_eval, t, y_t, p_t)
        recall0_tt = recalls_tt[:, 0]
        recall1_tt = recalls_tt[:, 1]
    return recall0_tt, recall1_tt, Gmean_tt


if __name__ == '__main__':
    theta = 1
    y = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    p = [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    [recall0, recall1, Gmean] = compute_online_PF(y, p, theta)

    print(recall0)
    print(recall1)
    print(Gmean)