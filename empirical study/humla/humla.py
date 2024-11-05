import os
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from itertools import product
from sklearn.metrics import mean_squared_error
from evaluate.evaluation_online import compute_online_PF
from humla.data_stream.real_data_stream import set_test_stream, set_train_stream
from data_stream.real_data_stream import data_id_2name
from humla.data_stream.real_data_stream import class_data_ind_org, class_data_ind_reset
from humla.data_stream.real_data_preprocess import real_data_preprocess
from humla.DenStream.DenStream import DenStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta.oza_bagging import OzaBaggingClassifier
from humla.core.oza_bagging_oob import OzaBaggingClassifier_OOB
from humla.core.oza_bagging_ooc import OzaBaggingClassifier_OOC
import pickle as pkl
import warnings
# auto para
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from sklearn.metrics import silhouette_score, recall_score
# lookup para_best based on our 14 projects in IJCNN'22
from humla.data_stream.real_data_stream import lookup_best_para
from humla.utility import load_para_csv, check_random_state, cvt_day2timestamp, cvt_timestamp2day
import matplotlib.pyplot as plt

# global variables
invalid_val, label_val = -1, [0, 1]
my_eps = np.finfo(np.float32).eps
dir_rslt_save = "./results/rslt.save/"
dir_plot_root = "./results/rslt.plot/"


def study_human_over_time(
        project_id=0, wait_days=15, clf_name="oob", nb_para_tune=500, nb_test=10000, seed_lst=range(100)):
    """
    Plot ave PFs with vs without human given that all class-1 predicted test instances are human labeled without noise.
    Updated on 2022/10/10, 11/9, 11/17
    """
    # setup
    save_plot = True
    verbose_int, is_plot, just_run = 0, False, False  # a_sdp_runs()
    to_dir_png = dir_plot_root + "pfs_cmp_test_steps/%s_%ddays_%dsd/" % (clf_name, wait_days, len(seed_lst))

    """benchmark, no human"""
    human_dict = {"has_human": False}
    retn_test_dct, retn_train_dct = a_sdp_runs(
        clf_name, human_dict, project_id, nb_para_tune, nb_test, wait_days, None,
        seed_lst, verbose_int, is_plot, just_run)
    gmean_tt_dct, r1_tt_dct, r0_tt_dct = retn_test_dct["gmean"], retn_test_dct["r1"], retn_test_dct["r0"]
    mcc_tt_dct, prec_tt_dct, f1_tt_dct = retn_test_dct["mcc"], retn_test_dct["precision"], retn_test_dct["f1_score"]

    """with_human at none human labeling noise and fully human labeling effort"""
    human_dict_has = {"has_human": True, "human_err": 0, "human_eff": 1}  # input
    clf_name_human = clf_name + "_human-err%s-eff%s" % (
        str(human_dict_has["human_err"]), str(human_dict_has["human_eff"]))
    retn_test_has_dct, retn_train_has_dct = a_sdp_runs(
        clf_name, human_dict_has, project_id, nb_para_tune, nb_test, wait_days, None,
        seed_lst, verbose_int, is_plot, just_run)
    gmean_tt_has_dct = retn_test_has_dct["gmean"]
    r1_tt_has_dct, r0_tt_has_dct = retn_test_has_dct["r1"], retn_test_has_dct["r0"]
    mcc_tt_has_dct = retn_test_has_dct["mcc"]
    prec_tt_has_dct, f1_tt_has_dct = retn_test_has_dct["precision"], retn_test_has_dct["f1_score"]

    """plotting: pf metrics"""
    info_pre = "%s-%s-%ddays-ave%d-" % (clf_name, data_id_2name(project_id), wait_days, len(seed_lst))
    clf_lst = (clf_name, clf_name_human)
    # plot for comparing
    pf_tt_dct_lst = [(gmean_tt_dct, gmean_tt_has_dct),
                     (r1_tt_dct, r1_tt_has_dct),
                     (r0_tt_dct, r0_tt_has_dct),
                     (mcc_tt_dct, mcc_tt_has_dct),
                     (prec_tt_dct, prec_tt_has_dct),
                     (f1_tt_dct, f1_tt_has_dct)]
    pf_name_lst = ["gmean", "r1", "r0", "mcc", "prec", "f1_score"]
    for pf_, pf_tt_dct_tuple_ in enumerate(pf_tt_dct_lst):
        pf_name_ = pf_name_lst[pf_].lower()
        if pf_name_ == "mcc".lower():
            my_ylim = (-1, 1)
        else:
            my_ylim = None
        pf_tt_ave_, pf_tt_has_ave_ = pf_tt_dct_tuple_[0]["ave"], pf_tt_dct_tuple_[1]["ave"]
        pf_np_ = np.column_stack((pf_tt_ave_, pf_tt_has_ave_))
        plot_on_1pf(pf_np_, clf_lst, info_pre + pf_name_, save_plot, to_dir_png + pf_name_ + "/", my_ylim)
    # print("Succeed in b_pfs_throughout_time() on %s" % datetime.today())


def auto_para_clf(project_id=0):
    """
    Conduct automatic parameter tuning of classifier for project_id.
    2024-4 create based on a_sdp_runs()
    """
    clf_name = "our".lower()
    wait_days = 15
    human_dict = {'has_human': False}  # para_tune based on the waiting-time labeling method
    nb_para_tune_clf = 1000
    nb_test = nb_para_tune_clf

    # input arguments for para_tune_clf
    is_plot = False
    is_print = False
    just_run = False  # False: do not store the found auto-para

    # ######################################################
    #
    # The below code was originally copied from a_sdp_runs()
    #
    # #######################################################
    assert human_dict is not None, "Have to assign human_dict"
    # auto distinguish RQ1 from (RQ2 & RQ3) based on keys
    if is_waiting_method(human_dict):  # waiting-time method in RQ1
        human_analyzer = HumanDictAnalyzer(human_dict)
    elif is_human_dict_RQ1(human_dict):  # for RQ1
        human_analyzer = HumanDictAnalyzer(human_dict)
    elif is_humla_study_RQ23(human_dict):  # RQ2 and RQ3
        human_analyzer = HumLaStudyAnalyzer(human_dict)
    else:
        raise Exception("Error input human_dict")

    # handle other inputs
    project_name = data_id_2name(project_id)
    info_run = "%s: %s, wtt=%d" % (clf_name, project_name, wait_days)

    """prepare test data_stream stream"""
    test_stream = set_test_stream(project_name)
    X_org = test_stream.X[class_data_ind_org().id_X_np]
    # convert #fea14 to #fea12 and the test data_stream stream
    XX_aug, use_data = real_data_preprocess(X_org)  # X_aug = [X_trans, churn_np]
    yy = test_stream.y[use_data]
    time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
    vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
    #
    n_data_all, n_fea = XX_aug.shape[0], XX_aug.shape[1] - 1  # X_aug = [X_trans, churn_np]
    assert n_fea == 12, "# transformed #fea should be 12. Sth. must be wrong."

    # fea normalizer based on all test data_stream, which is used for DenStream
    norm_scaler = my_norm_scaler(n_fea=n_fea, norm_name="z_score")
    norm_scaler.my_fit(XX_aug[:, :-1])  # the last col is churn_np
    # print('std:', np.std(norm_scaler.my_transform(XX), axis=0))

    # prepare all test samples
    test_data_all = np.hstack((time, XX_aug, vl, yy))  # col=3+1+12 ~ (time, #fea12, churn, vl, yy)
    # judge if the project if longer enough than 1w, added on 2024/4/9
    nb_test_has = test_data_all.shape[0]
    nb_test_act = nb_test - nb_para_tune_clf
    if nb_test_has < nb_test_act:
        raise Exception("The code should not be run for project=%d (%s) since its length=%d < nb_test=%d"
                        % (project_id, project_name, nb_test_has, nb_test_act))
    # data index reset
    data_ind_reset = class_data_ind_reset(
        id_time=0, id_y=-1, id_vl=-2, id_churn=-3, id_X_np=np.s_[:, 1:1 + n_fea], n_fea=n_fea)

    # data_stream pre-train
    data_ptrn = test_data_all[:nb_para_tune_clf]  # (time, #fea12, churn, vl, y)
    X_ptrn, y_ptrn = data_ptrn[data_ind_reset.id_X_np], data_ptrn[:, data_ind_reset.id_y]

    """para of DenStream~(lambda, eps, beta, mu)"""
    our_clf_lst = ("our", "our_others")  # vip manually maintain
    if any(clf_name == clf_ for _, clf_ in enumerate(our_clf_lst)):
        X_ptrn_norm = norm_scaler.my_transform(X_ptrn)
        eps, mu, beta, lambd = 1.47, 1.57, 0.78, 0.26  # shuxian tuned during odasc++
        # eps, mu, beta, lambd = 2.09, 2.20, 0.74, 0.125
        # lambd, eps, beta, mu = 0.1, 1.5, 0.5, 3  # shuxian default in ijcnn'22

    """pre-train DenStream"""
    if any(clf_name == clf_ for _, clf_ in enumerate(our_clf_lst)):
        cluster = DenStream(theta_cl=None, lambd=lambd, eps=eps, beta=beta, mu=mu)
        cluster.partial_fit(X_ptrn_norm, y_ptrn)
        # plot 2d-pca
        if is_plot:
            x_lim, y_lim = None, None
            all_yy = test_data_all[:, data_ind_reset.id_y]
            all_X_org = test_data_all[data_ind_reset.id_X_np]
            all_X_norm = norm_scaler.my_transform(all_X_org)

            # conduct pca on all points
            pca_hd = PCA(n_components=2)
            all_X_norm_pca = pca_hd.fit_transform(all_X_norm)
            print("%s: pca 2d output >>>" % info_run)
            print("\tpca.explained_var_ratios are ", pca_hd.explained_variance_ratio_)
            print("\tpca.singular_values are ", pca_hd.singular_values_)
            # 2d pca scatter of all points
            plt.scatter(all_X_norm_pca[:, 0], all_X_norm_pca[:, 1], c=all_yy)
            plt.title("%s: 2D scatter of normed pca points" % project_name)
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.grid(True)
            plt.show()
            # 2d micro-clusters on pre-train samples
            cluster.plot_cluster(X_ptrn_norm, y_ptrn, pca_hd, "%s: data_stream pre-train" % project_name, x_lim, y_lim)
    else:
        cluster = None  # for auto_classifier for oob and oza

    """auto-para classifiers~(n_tree, theta_imb, theta_cl)"""
    dir_auto_para = dir_rslt_save + "auto_para/%s-%dd/" % (clf_name, wait_days)
    os.makedirs(dir_auto_para, exist_ok=True)
    auto_name = "%s-bst_para-T%d" % (data_id_2name(project_id), nb_para_tune_clf) + ".pkl"
    exist_clf_para = os.path.exists(dir_auto_para + auto_name)  # check existence
    # para search
    if exist_clf_para and not just_run:  # load para_opt
        para_dict = pkl.load(open(dir_auto_para + auto_name, 'rb'))
        n_tree, theta_imb, theta_cl = para_dict["n_trees"], para_dict["theta_imb"], para_dict["theta_cl"]
    else:  # search para_opt; take time
        if is_print:
            print("%s: auto para-tuning, taking time..." % info_run)
        n_tree, theta_imb, theta_cl = para_classifier(cluster, clf_name, X_ptrn, y_ptrn)  # taking time
        if not just_run:  # save para_bst
            para_dict = {"n_trees": n_tree, "theta_imb": theta_imb, "theta_cl": theta_cl}
            with open(dir_auto_para + auto_name, 'wb') as save_file:
                pkl.dump(para_dict, save_file)
    if is_print:
        print("%s, best para:\n\tn_test=%d, n_tree=%d, theta_imb=%.3f, theta_cl=%.3f" % (
            info_run, nb_test, n_tree, theta_imb, theta_cl))


def example_sdp_clf(example_id=0, project_id=0, just_run=True):
    # This script is to demonstrate examples of the structure human_dict
    # Liyan on 2024/4/12
    isinstance(example_id, int)
    if example_id == 0:  # the waiting-time method
        print("run the waiting-time method")
        human_dict = {'has_human': False}
    elif example_id == 1:  # HumLa
        print("run HumLa")
        human_dict = {'has_human': True, 'human_err': 0, 'human_eff': 1}
    elif example_id == 2:  # eco-humla
        print("run Eco-HumLa")
        human_dict = {'has_human': True, 'human_err': 0, 'human_eff': 'auto'}
    else:
        raise Exception("Error in example_id=%d" % example_id)

    # run jit_sdp
    clf_name, nb_para_tune, nb_test, wait_days = 'our', 1000, 10000, 15
    a_sdp_runs(clf_name, human_dict, project_id, nb_para_tune, nb_test, wait_days, is_plot=False, just_run=just_run)


def a_sdp_runs(clf_name="our", human_dict=None,
               project_id=0, nb_para_tune=500, nb_test=10000, wait_days=30,
               para_csv=None, seed_lst=range(10), verbose_int=2, is_plot=False, just_run=False, just_load=False):
    """
    Inputs:
        @human_dict: see the usage in example_run_sdp()
        @verbose_int:
            * larger values mean deeper and more "print".
            * -1 no print at all
        @just_run: Bool
            * True: just run for investigation, not save anything for safety reasons.
        @just_load: Bool
            * True: just load existing results, not run or save anything for safety reasons.
    Liyan Song
    2022/8/8    re-configurate the code
    2022/10/10  add has_human
    2023/11/9   reformat the return variables to store PFs over all seeds to conduct anova (across seeds within group),
                consequently, functions in xRQs.py and xRQs_post.py may need to update to be able to call
                this main function.
    Updated for HumLa-extension on 2023/11/13
    """

    assert human_dict is not None, "Have to assign human_dict"
    # auto distinguish RQ1 from (RQ2 & RQ3) based on keys
    if is_waiting_method(human_dict):  # waiting-time method in RQ1
        human_analyzer = HumanDictAnalyzer(human_dict)
    elif is_human_dict_RQ1(human_dict):  # for RQ1
        human_analyzer = HumanDictAnalyzer(human_dict)
    elif is_humla_study_RQ23(human_dict):  # RQ2 and RQ3
        human_analyzer = HumLaStudyAnalyzer(human_dict)
    else:
        raise Exception("Error input human_dict")

    # handle other inputs
    clf_name = clf_name.lower()
    if para_csv is None:
        para_csv = load_para_csv()
    project_name = data_id_2name(project_id)
    # print title info
    info_run = "%s: %s, wtt=%d, #seed=%d" % (clf_name, project_name, wait_days, len(seed_lst))
    if human_analyzer.has_human:  # overwritten
        info_run += ", has_human: error=%s, effort=%s" % (
            str(human_analyzer.human_err), str(human_analyzer.human_eff))

    #######################################################################
    #
    # Module: set up the streaming data
    #
    #######################################################################
    # prepare test data_stream stream
    test_stream = set_test_stream(project_name)
    X_org = test_stream.X[class_data_ind_org().id_X_np]
    # convert #fea14 to #fea12 and the test data_stream stream
    XX_aug, use_data = real_data_preprocess(X_org)  # X_aug = [X_trans, churn_np]
    yy = test_stream.y[use_data]
    time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
    vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]

    # handle negative nb_test
    n_data_all, n_fea = XX_aug.shape[0], XX_aug.shape[1] - 1  # X_aug = [X_trans, churn_np]
    assert n_fea == 12, "# transformed #fea should be 12. Sth. must be wrong."
    if nb_test < 0:
        nb_test += n_data_all
        if verbose_int >= 2:
            print("actual nb_test=%d" % nb_test)
    assert nb_para_tune < nb_test, "nb_pre=%d should be smaller than nb_data=%d" % (nb_para_tune, nb_test)

    # fea normalizer based on all test data_stream, which is used for DenStream
    norm_scaler = my_norm_scaler(n_fea=n_fea, norm_name="z_score")
    norm_scaler.my_fit(XX_aug[:, :-1])  # the last col is churn_np
    # print('std:', np.std(norm_scaler.my_transform(XX), axis=0))

    # prepare all test samples
    test_data_all = np.hstack((time, XX_aug, vl, yy))  # col=3+1+12 ~ (time, #fea12, churn, vl, yy)
    # judge if the project if longer enough than 1w, added on 2024/4/9
    nb_test_has = test_data_all.shape[0]
    nb_test_act = nb_test - nb_para_tune
    if nb_test_has < nb_test_act:
        raise Exception("The code should not be run for project=%d (%s) since its length=%d < nb_test=%d"
                        % (project_id, project_name, nb_test_has, nb_test_act))
    # data index reset
    data_ind_reset = class_data_ind_reset(
        id_time=0, id_y=-1, id_vl=-2, id_churn=-3, id_X_np=np.s_[:, 1:1 + n_fea], n_fea=n_fea)

    # data_stream pre-train
    data_ptrn = test_data_all[:nb_para_tune]  # (time, #fea12, churn, vl, y)
    X_ptrn, y_ptrn = data_ptrn[data_ind_reset.id_X_np], data_ptrn[:, data_ind_reset.id_y]
    # churn_ptrn = data_ptrn[:, data_ind_reset.id_churn]

    ###########################################################################################
    # 2024-04 New
    # auto para: DenStream and Classifier using the waiting=time labeling method
    ###########################################################################################
    """auto-para DenStream~(lambda, eps, beta, mu)"""
    our_clf_lst = ("our", "our_others")  # vip manually maintain
    if any(clf_name == clf_ for _, clf_ in enumerate(our_clf_lst)):
        X_ptrn_norm = norm_scaler.my_transform(X_ptrn)
        auto_denStream = False
        if auto_denStream:
            eps, mu, beta, lambd = para_denStream(X_ptrn_norm, y_ptrn, nb_repeat=10)
        else:  # 2023/11 check current hyper-para of DenStream after #fea12, still OK? Yes 2024/4
            eps, mu, beta, lambd = 1.47, 1.57, 0.78, 0.26  # shuxian tuned during odasc++
            # eps, mu, beta, lambd = 2.09, 2.20, 0.74, 0.125
            # lambd, eps, beta, mu = 0.1, 1.5, 0.5, 3  # shuxian default in ijcnn'22

    """pre-train DenStream"""
    if any(clf_name == clf_ for _, clf_ in enumerate(our_clf_lst)):
        cluster = DenStream(theta_cl=None, lambd=lambd, eps=eps, beta=beta, mu=mu)
        cluster.partial_fit(X_ptrn_norm, y_ptrn)
        # plot 2d-pca
        if is_plot:
            x_lim, y_lim = None, None
            all_yy = test_data_all[:, data_ind_reset.id_y]
            all_X_org = test_data_all[data_ind_reset.id_X_np]
            all_X_norm = norm_scaler.my_transform(all_X_org)

            # conduct pca on all points
            pca_hd = PCA(n_components=2)
            all_X_norm_pca = pca_hd.fit_transform(all_X_norm)
            print("\t pca 2d output:")
            print("\t\tpca.explained_var_ratios are ", pca_hd.explained_variance_ratio_)
            print("\t\tpca.singular_values are ", pca_hd.singular_values_)
            # 2d pca scatter of all points
            plt.scatter(all_X_norm_pca[:, 0], all_X_norm_pca[:, 1], c=all_yy)
            plt.title("%s: 2D scatter of normed pca points" % project_name)
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.grid(True)
            plt.show()
            # 2d micro-clusters on pre-train samples
            cluster.plot_cluster(X_ptrn_norm, y_ptrn, pca_hd, "%s: data_stream pre-train" % project_name, x_lim, y_lim)
    else:
        cluster = None  # for auto_classifier for oob and oza and pbsa

    """auto-para classifiers~(n_tree, theta_imb, theta_cl)"""
    # Log 2024/04/08    setup for individual project. This suites to RQ1~RQ3.
    # if project_id in range(0, 14):  # previous projects
    #     auto_clf = False  # load
    # elif project_id in range(14, 23):  # new projects
    #     auto_clf = True  # run, will take time
    # else:
    #     raise Exception("Undefined project_id=%d." % project_id)
    # if not auto_clf:  # load
    #     # n_tree, theta_imb, theta_cl = 5, 0.95, 0.8  # manual setup
    #     # the 14 projects that had been para-tuned in ODaSC [ijcnn'22] and Humla [fse'23]
    #     n_tree, theta_imb, theta_cl = lookup_best_para(project_name, wait_days, clf_name, para_csv)
    # else:  # run, taking time - not in use 2022 summer holiday
    dir_auto_para = dir_rslt_save + data_id_2name(project_id)+"/%s-%dd/" % (clf_name, wait_days)
    os.makedirs(dir_auto_para, exist_ok=True)
    nb_para_clf = 1000
    auto_name = "%s-bst_para-T%d" % (data_id_2name(project_id), nb_para_clf) + ".pkl"
    exist_clf_para = os.path.exists(dir_auto_para + auto_name)  # check existence
    # para search
    if exist_clf_para and not just_run:  # load para_opt
        para_dict = pkl.load(open(dir_auto_para + auto_name, 'rb'))
        n_tree, theta_imb, theta_cl = para_dict["n_trees"], para_dict["theta_imb"], para_dict["theta_cl"]
    else:  # search para_opt; take time
        # raise Exception("TMP stop auto-para-clf to avoid error in multi-run. 2024-4")
        # 2024-4 TMP comment off to avoid defects in multi-processing.
        if verbose_int >= 1:
            print("%s: auto para-tuning and it takes time..." % info_run)
        n_tree, theta_imb, theta_cl = para_classifier(cluster, clf_name, X_ptrn, y_ptrn)  # take time
        if not just_run:  # save para_bst
            para_dict = {"n_trees": n_tree, "theta_imb": theta_imb, "theta_cl": theta_cl}
            with open(dir_auto_para + auto_name, 'wb') as save_file:
                pkl.dump(para_dict, save_file)
    if verbose_int >= 1:
        print("\t%s, best para:\n\t\tn_test=%d, n_tree=%d, theta_imb=%.3f, theta_cl=%.3f" % (
            info_run, nb_test, n_tree, theta_imb, theta_cl))

# update DenStream para
    if any(clf_name == clf_ for _, clf_ in enumerate(our_clf_lst)):
        cluster.theta_cl = theta_cl

    # ###########################################################################
    #
    # The main run in loop across random seeds
    #
    # ###########################################################################
    nb_train_delay_np, nb_train_human_np = np.empty((len(seed_lst))), np.empty((len(seed_lst)))
    acc_train_churn_np = np.empty((len(seed_lst)))  # init
    nb_pred_y_np = np.empty((len(seed_lst), 2))
    for ss, seed in enumerate(seed_lst):
        if is_human_dict_RQ1(human_dict) or is_waiting_method(human_dict):
            to_dir = rslt_dir_RQ1(
                clf_name, human_dict, project_id, wait_days, n_tree, theta_imb, theta_cl)
        elif is_humla_study_RQ23(human_dict):
            to_dir = rslt_dir_RQ23(
                clf_name, human_dict, project_id, wait_days, n_tree, theta_imb, theta_cl)
        os.makedirs(to_dir, exist_ok=True)  # keep this
        # analyze filenames of to_dir: load the results of "T > nb_data" to save computational cost.
        exist_result, to_dir, nb_test_saved = rslt_dir_analyze(to_dir, clf_name, nb_test, seed)
        # if just_load: abort here
        if not exist_result and just_load:  # only load existing results & Do not run any new results
            raise Exception("Merely existing results are loaded, which do NOT exist.")
        # if not just_load: we can run to get new results
        if not exist_result:
            to_dir += "/T" + str(nb_test) + "/"
            os.makedirs(to_dir, exist_ok=True)
        # file_name-s
        flnm_test = "%s%s.rslt_test.s%d" % (to_dir, clf_name, seed)
        flnm_train = "%s%s.rslt_train.s%d" % (to_dir, clf_name, seed)

        """load or compute the result for each seed"""
        if exist_result and not just_run:  # load
            rslt_test = np.loadtxt(flnm_test)
            rslt_train = np.loadtxt(flnm_train)

            # for the test set: starting and ending points
            nb_para_tune_saved = nb_test_saved - rslt_test.shape[0]
            nb_test_act = nb_test - nb_para_tune
            pos_stt = nb_para_tune - nb_para_tune_saved
            assert pos_stt >= 0, \
                "error: cannot fetch results from the case that has smaller tuning steps than this running case."
            pos_end = pos_stt + nb_test_act
            # extract the results if len(rslt_test) > nb_test_actual. Refer to the computational case.
            if len(rslt_test) > nb_test_act:
                rslt_test = rslt_test[pos_stt:pos_end, :]
                id_test_time = 0
                test_stt_time, test_end_time = rslt_test[0, id_test_time], rslt_test[-1, id_test_time]
                # for the train set:
                id_train_use_time = 1
                pos_trn_stt = np.where(test_stt_time <= rslt_train[:, id_train_use_time])[0][0]
                pos_trn_end = np.where(rslt_train[:, id_train_use_time] <= test_end_time)[0][-1]
                rslt_train = rslt_train[pos_trn_stt:pos_trn_end + 1, :]
        else:  # compute, taking time
            if human_analyzer.has_human:
                my_rng = check_random_state(seed)
            """pre-train classifier"""
            if clf_name == "oza":
                classifier = OzaBaggingClassifier(HoeffdingTreeClassifier(), n_tree, seed)
                classifier.partial_fit(X_ptrn, y_ptrn, label_val)
            elif clf_name == "oob":  # oob: (theta_imb, n_trees)
                classifier = OzaBaggingClassifier_OOB(HoeffdingTreeClassifier(), n_tree, seed, theta_imb)
                classifier.partial_fit(X_ptrn, y_ptrn, label_val)
            elif clf_name == "oob_filter".lower():  # filter out noisy data_stream
                classifier = OzaBaggingClassifier_OOB(HoeffdingTreeClassifier(), n_tree, seed, theta_imb)
                cl_ptrn = comp_cl_upper(y_ptrn, y_ptrn)  # suppose noise-free
                use_bool = cl_ptrn == 1  # filter
                classifier.partial_fit(X_ptrn[use_bool], y_ptrn[use_bool], label_val)
            elif clf_name == "our":  # our: (theta_imb, n_trees, theta_cl)
                classifier = OzaBaggingClassifier_OOC(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, theta_cl)
                cl_ptrn = comp_cl_upper(y_ptrn, y_ptrn)  # suppose noise-free
                classifier.partial_fit(X_ptrn, y_ptrn, cl_ptrn, label_val)
            elif clf_name == "our_filter".lower():  # filter out noisy data_stream
                classifier = OzaBaggingClassifier_OOC(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, theta_cl)
                cl_ptrn = comp_cl_upper(y_ptrn, y_ptrn)  # suppose noise-free
                use_bool = cl_ptrn == 1  # filter
                classifier.partial_fit(X_ptrn[use_bool], y_ptrn[use_bool], cl_ptrn[use_bool], label_val)
            elif clf_name == "our_upp".lower():  # upperbound of ODaSC, may NOT be a good competitor
                classifier = OzaBaggingClassifier_OOC(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, theta_cl)
                cl_ptrn = comp_cl_upper(y_ptrn, y_ptrn)  # suppose noise-free
                classifier.partial_fit(X_ptrn, y_ptrn, cl_ptrn, label_val)
            elif clf_name == "our_btm":
                classifier = OzaBaggingClassifier_OOC(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, theta_cl)
                rng = check_random_state(seed)
                cl_ptrn = rng.uniform(0, 1, len(y_ptrn))
                classifier.partial_fit(X_ptrn, y_ptrn, cl_ptrn, label_val)
            else:
                raise Exception("Undefined clf_name=%s. Existing clf_names include %s"
                                % (clf_name, "oza, oob, oob_filter, our, our_filter, our_upp, our_btm,pbsa"))

            """[core] test-then-train:
            at each test step, only 1 test data arrives; none or several training data would become available.
            """
            # init, test stream related. Note: we can know the size in advance.
            test_time, test_y_tru, test_y_pre = np.empty(nb_test_act), np.empty(nb_test_act), np.empty(nb_test_act)
            on_imb0, on_imb1 = invalid_val * np.ones(nb_test_act), invalid_val * np.ones(nb_test_act)
            # init, train stream related. Note that we cannot know the size cannot in advance.
            cmt_time_train_lst, use_time_train_lst, y_train_tru_lst, y_train_obv_lst = [], [], [], []
            code_churn_lst, cl_train_lst, use_cluster_lst = [], [], []
            # init, the test process
            prev_test_time = data_ptrn[-1, data_ind_reset.id_time]  # vip
            data_buffer, nb_train_, nb_train_human_ = np.empty((0, data_ptrn.shape[1])), 0, 0

            # for each test step
            for tt in range(nb_test_act):
                # get the test data_stream
                test_step = tt + nb_para_tune
                test_1data = test_data_all[test_step, :].reshape((1, -1))  # 16~(time, #fea12, churn, vl, yy)
                test_X, test_churn = test_1data[data_ind_reset.id_X_np], test_1data[0, data_ind_reset.id_churn]
                test_time[tt] = test_1data[:, data_ind_reset.id_time]
                test_y_tru[tt] = test_1data[:, data_ind_reset.id_y]

                """test: predict with classifiers"""
                test_y_pre[tt] = classifier.predict(test_X)[0]

                """[core] ext-HumLa for RQ1~RQ3 -- handle human-effort"""
                new_1data = test_1data  # overwritten if correct human labeling. VIP
                sota_X, sota_churn, sota_time, sota_y_obv, sota_y_tru = \
                    np.empty((0, n_fea)), np.empty(0), np.empty(0), np.empty(0), np.empty(0)  # init empty, required
                if human_analyzer.has_human and test_y_pre[tt] == 1:
                    if is_human_dict_RQ1(human_dict):  # RQ1
                        if human_analyzer.is_humla():  # RQ1.humla
                            prob_human_effort = human_analyzer.human_eff  # fixed for all test data
                        elif human_analyzer.is_eco_humala():  # RQ1.eco-humla
                            y_pre_prob_ = classifier.predict_proba(test_X)[0]
                            y_pre_diff = y_pre_prob_[1] - y_pre_prob_[0]  # as y_pre==1, y_pre_prob_ > 0 must hold
                            assert 0 <= y_pre_diff <= 1, "Sth must be wrong. PLS debug."
                            # choose one of the following options
                            my_auto_ = "auto_most"
                            if my_auto_ == "auto_least":  # not in use
                                prob_human_effort = 1 - y_pre_diff  # the least confident and most informative case
                            elif my_auto_ == "auto_most":  # Eco-HumLa
                                prob_human_effort = y_pre_diff  # the most confident and least informative case
                            else:
                                raise "Error in human_eff=%s for Eco_HumLa" % my_auto_
                            assert 0 <= prob_human_effort <= 1, "Sth must be wrong. PLS debug."
                    elif is_humla_study_RQ23(human_dict):  # for RQ2 and RQ3
                        if human_analyzer.is_study_RQ2_vl_humla():  # RQ2 - humla
                            assert human_analyzer.human_eff == get_human_random_name(), \
                                "in RQ2 for humla, we should have 'random' human effort setting"
                            prob_human_effort = my_rng.uniform(0, 1)  # random human noise
                            if verbose_int >= 1:
                                print("\tin RQ2 for HumLa: random prob_human_effort=%f" % prob_human_effort)
                        elif human_analyzer.is_study_RQ2_vl_eco():  # RQ2 - eco-humla
                            assert human_analyzer.human_eff == get_human_auto_name(), \
                                "eco-humla should have 'auto' human effort setting"
                            # copied codes hereafter
                            y_pre_prob_ = classifier.predict_proba(test_X)[0]
                            y_pre_diff = y_pre_prob_[1] - y_pre_prob_[0]  # as y_pre==1, y_pre_prob_ > 0 must hold
                            assert 0 <= y_pre_diff <= 1, "Sth must be wrong. PLS debug."
                            prob_human_effort = y_pre_diff  # the most confident and least informative case
                            assert 0 <= prob_human_effort <= 1, "Sth must be wrong. PLS debug."
                        elif human_analyzer.is_study_RQ3_eco_least():  # RQ3 - eco-humla exclusively
                            assert human_analyzer.human_eff == get_human_auto_name(), \
                                "eco-humla should have 'auto' human effort setting"
                            # copied codes hereafter: investigate the most vs least confident defect predictions
                            y_pre_prob_ = classifier.predict_proba(test_X)[0]
                            y_pre_diff = y_pre_prob_[1] - y_pre_prob_[0]  # as y_pre==1, y_pre_prob_ > 0 must hold
                            assert 0 <= y_pre_diff <= 1, "Sth must be wrong. PLS debug."
                            prob_auto_least = 1 - y_pre_diff  # the least confident yet most informative case
                            prob_auto_most = y_pre_diff  # eco-humla: the most confident yet least informative case
                            # [core] the most-vs-least confident prediction
                            this_random_ = my_rng.uniform(0, 1)
                            if this_random_ <= human_analyzer.get_RQ3_eco_prob_least():
                                prob_human_effort = prob_auto_least  # the opposite case
                            else:
                                prob_human_effort = prob_auto_most  # eco-humla
                    else:
                        raise Exception("Error here: Liyan on Nov.2023.")

                    """[core] HumLa-ext: RQ1~RQ3 -- human labeling noise. 
                    The human label noise is 1-sided. Given y_pre = 1,
                        * if y_tru = 0: y_human = 0
                        * if y_tru = 1: y_human may be 0 (wrong) or 1 (correct) 
                    """
                    if my_rng.uniform(0, 1) <= prob_human_effort:  # do HumLa (Eco-HumLa)
                        # overwrite empty when correct human labeling; note np.array([...]), vip
                        sota_X, sota_churn, sota_time, sota_y_tru = \
                            test_X, np.array([test_churn]), np.array([test_time[tt]]), np.array([test_y_tru[tt]])
                        # probability of committing 1-sided human labeling noise
                        if sota_y_tru[0] == 1:  # for truly defective data
                            if is_human_dict_RQ1(human_dict):  # for RQ1
                                prob_human_error = human_analyzer.human_err
                            elif is_humla_study_RQ23(human_dict):  # for RQ2 and RQ3
                                if human_analyzer.human_err == get_human_random_name():
                                    prob_human_error = my_rng.uniform(0, 1)  # random human label noise
                                else:
                                    raise Exception("Error in human_error for RQ2 or RQ3.")
                            # decide the occurrence of human labeling error stochastically
                            if my_rng.uniform(0, 1) <= 1 - prob_human_error:  # correct human labeling
                                sota_y_obv = np.array([test_y_tru[tt]])  # note np.array([...])
                                new_1data = np.empty((0, data_ptrn.shape[1]))  # overwrite as empty not store to buffer
                            else:  # wrong human labeling
                                sota_y_obv = np.array([1 - test_y_tru[tt]])
                        elif sota_y_tru[0] == 0:  # for truly clean data, no label noise
                            sota_y_obv = np.array([test_y_tru[tt]])  # note np.array([...])
                            new_1data = np.empty((0, data_ptrn.shape[1]))  # overwrite as empty not store to buffer
                        else:
                            raise Exception("the class label should be 0/1 exclusively.")

                        # RQ2 exclusively (HumLa and Eco-HumLa):
                        #   For non-zero human-VL-hour, run the below;
                        #   for zero humla-vl-hour, no need to run.
                        # 2024/03/21 RQ2-vl Only relate to the below codes
                        # 2024/04/08 debug, 2024-07-25 debug todo
                        if is_human_dict_RQ1(human_dict):  # for RQ1
                            human_vl_day = 0  # RQ1 supposes zero human vl
                        elif is_humla_study_RQ23(human_dict):  # RQ2 and RQ3
                            human_vl_day = human_analyzer.get_RQ2_vl_hour() / 24  # 1day=24h
                            # [debug] 2024-07-25 Liyan: finally, this tiny error!!!
                            if human_vl_day == 0:  # reset to avoid code defect
                                human_vl_day = my_eps  # set_train_stream() will reset [vl=0] to [vl=inf]
                        if human_vl_day > 0 and sota_X.size > 0:
                            # 1) assign sota_data and its human_VL_hour to new_1data~(time, #fea12, churn, vl, yy) #=16
                            human_data_vl = np.column_stack(
                                [sota_time, sota_X, sota_churn, human_vl_day, sota_y_obv])
                            # Liyan on 2023/12/22 debug
                            test_1data_vl = test_1data[0, data_ind_reset.id_vl]  # overwritten maybe below
                            if test_1data_vl > 0:  # avoid that clean with its vl=0 incorrectly (and truly defect only)
                                if test_1data_vl < human_vl_day:  # overwrite human_data_vl to empty
                                    human_data_vl = np.empty((0, data_ptrn.shape[1]))
                            # vip when & only when perform human labeling and the labeling is incorrect, #new_1data=2
                            # this criterion will be used by data_buffer in set_train_stream()
                            #   Complement on 2024-07-25, Liyan, vvip
                            new_1data = np.vstack((new_1data, human_data_vl))
                            # 2) reset sota_data to empty if human-vl exists
                            sota_X, sota_churn, sota_time, sota_y_obv, sota_y_tru = \
                                np.empty((0, n_fea)), np.empty(0), np.empty(0), np.empty(0), np.empty(0)

                """get the new train data_stream batch:
                Liyan on 2024-7-25 complement: the following defect and clean training data production is correct 
                    based on the strict rule on data_buffer implemented in set_train_stream().
                    Specifically, we require for data_buffer that: the vl must be genuine, which can be easily met
                    previously given that human vl is by default zero, for RQ1 and RQ3. 
                    However, when it comes to RQ2, things become much more complex!
                """
                data_buffer, new_train_def, new_train_cln, new_train_unl = set_train_stream(
                    prev_test_time, test_time[tt], new_1data, data_ind_reset, data_buffer, wait_days)
                # produce new training data. Note the order (clean, defect, sota)
                cmt_time_train = np.concatenate((
                    new_train_cln[:, data_ind_reset.id_time], new_train_def[:, data_ind_reset.id_time], sota_time))
                use_time_train = test_time[tt] * np.ones(cmt_time_train.shape)
                X_train = np.concatenate((
                    new_train_cln[data_ind_reset.id_X_np], new_train_def[data_ind_reset.id_X_np], sota_X))
                y_train_obv = np.concatenate((
                    np.zeros(new_train_cln.shape[0]), np.ones(new_train_def.shape[0]), sota_y_obv))
                y_train_tru = np.concatenate((
                    new_train_cln[:, data_ind_reset.id_y], new_train_def[:, data_ind_reset.id_y], sota_y_tru))
                churn_train = np.concatenate((
                    new_train_cln[:, data_ind_reset.id_churn], new_train_def[:, data_ind_reset.id_churn], sota_churn))
                # tmp counter
                nb_train_ += y_train_obv.shape[0]
                nb_train_human_ += sota_y_tru.shape[0]  # wrong 2024/3/21, almost impossible to count when human_vl>0
                # get class imbalance info: note put before classifier.partial_fit()
                if clf_name != "oza":
                    on_imb0[tt], on_imb1[tt] = classifier.rho0, classifier.rho1

                # assign
                cmt_time_train_lst.extend(cmt_time_train.tolist())
                code_churn_lst.extend(churn_train)
                use_time_train_lst.extend(use_time_train.tolist())
                y_train_obv_lst.extend(y_train_obv.tolist())
                y_train_tru_lst.extend(y_train_tru.tolist())
                if verbose_int >= 2:
                    print("\ttest_step=%d, y_true=%d, y_pre=%d: %s, has_human:%d"
                          % (test_step, test_y_tru[tt], test_y_pre[tt], clf_name, human_analyzer.has_human))
                    print("\t\tnew_train: y_true=%s, y_obv=%s" % (str(y_train_tru), str(y_train_obv)))
                    print("\t\t#acc_train_all=%d, #acc_train_human=%d" % (nb_train_, nb_train_human_))

                """update: classifiers and DenStream using the newly labelled training data_stream"""
                if y_train_obv.shape[0] > 0:
                    if clf_name == "oza" or clf_name == "oob" or clf_name=="pbsa":
                        classifier.partial_fit(X_train, y_train_obv, label_val)
                        # assign
                        cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                        use_cluster_lst = cl_train_lst
                    elif clf_name == "oob_filter":
                        cl_train = comp_cl_upper(y_train_tru, y_train_obv)
                        use_bool = cl_train == 1
                        classifier.partial_fit(X_train[use_bool], y_train_obv[use_bool], label_val)
                        # assign
                        cl_train_lst.extend(cl_train.tolist())
                        use_cluster_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                    elif clf_name == "our":
                        X_train_norm = norm_scaler.my_transform(X_train)
                        cl_train, cl_c1_refine, use_cluster_train = cluster.compute_CLs(X_train_norm, y_train_obv)
                        # update classifier
                        classifier.partial_fit(X_train, y_train_obv, cl_train, label_val)
                        # update micro-cluster
                        cluster.partial_fit(X_train_norm, y_train_obv)
                        cluster.revise_cluster_info(X_train_norm, y_train_obv, cl_train)
                        # assign
                        cl_train_lst.extend(cl_train.tolist())
                        use_cluster_lst.extend(use_cluster_train.tolist())
                        # print
                        if verbose_int >= 2:
                            for y_tru_, y_obv_, cl_ in zip(y_train_tru, y_train_obv, cl_train):
                                print("\t\t\ty_trn_tru=%d, y_trn_obv=%d, cl_est=%.2f" % (y_tru_, y_obv_, cl_))
                        if is_plot and False:  # manual control
                            info = "test-step=%d, train X_org with y_true" % test_step
                            cluster.plot_cluster(X_train_norm, y_train_tru, pca_hd, info, x_lim, y_lim, True)
                    elif clf_name == "our_filter":  # filter out noisy data_stream
                        cl_train = comp_cl_upper(y_train_tru, y_train_obv)
                        use_bool = cl_train == 1
                        classifier.partial_fit(X_train[use_bool], y_train_obv[use_bool], cl_train[use_bool], label_val)
                        # assign
                        cl_train_lst.extend(cl_train.tolist())
                        use_cluster_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                    elif clf_name == "our_upp":  # upper-bound of CL
                        cl_train = comp_cl_upper(y_train_obv, y_train_tru)
                        classifier.partial_fit(X_train, y_train_obv, cl_train, label_val)
                        # assign
                        cl_train_lst.extend(cl_train.tolist())
                        use_cluster_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                    elif clf_name == "our_btm":
                        cl_train = np.random.uniform(0, 1, len(y_train_obv))
                        classifier.partial_fit(X_train, y_train_obv, cl_train, label_val)
                        # assign
                        cl_train_lst.extend(cl_train.tolist())
                        use_cluster_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                    else:
                        raise Exception("Undefined classifier with clf_name=%s." % clf_name)
                prev_test_time = test_time[tt]  # update VIP

            """save returns"""
            # return 1: rslt_test ~ (test_time, y_true, y_pred)
            rslt_test = np.vstack((test_time, test_y_tru, test_y_pre)).T
            # return 2: rslt_train ~ (commit_time, use_time, yy, y_obv, cl, use_cluster, code_churn)
            cl_pre, use_cluster = np.array(cl_train_lst), np.array(use_cluster_lst)
            rslt_train = np.vstack((
                np.array(cmt_time_train_lst), np.array(use_time_train_lst),
                np.array(y_train_tru_lst), np.array(y_train_obv_lst),
                cl_pre, use_cluster, np.array(code_churn_lst))).T
            if not just_run and not just_load:  # save, 2024-4-16 debug the logic
                info_str = ". \tNote: '%d' means invalidity" % invalid_val
                np.savetxt(flnm_test, rslt_test, fmt='%d\t %d\t %d',
                           header="%test_time, yy, y_pre) " + info_str)
                np.savetxt(flnm_train, rslt_train, fmt='%d %d\t %d\t %d\t %f\t %d\t %.2f',
                           header="%commit_time, use_time, yy, y_obv, CL, use_cluster, code_churn" + info_str)

        """compute VIP statistics of stream_train"""
        id_train_human_np = np.where(rslt_train[:, 0] == rslt_train[:, 1])[0]
        nb_train_human_np[ss] = id_train_human_np.shape[0]
        nb_train_delay_np[ss] = rslt_train.shape[0] - nb_train_human_np[ss]
        id_train_churn = -1  # manual check
        acc_train_churn_np[ss] = np.nansum(rslt_train[id_train_human_np, id_train_churn])  # accumulated churn
        nb_pred_y_np[ss, 0], nb_pred_y_np[ss, 1] = np.sum(rslt_test[:, 2] == 0), np.sum(rslt_test[:, 2] == 1)
        if verbose_int >= 1:
            print("\n" + "--" * 20)
            print("%s - seed=%d: " % (info_run, seed))
            print("\t nb_train_delay=%d, nb_train_human=%d." % (nb_train_delay_np[ss], nb_train_human_np[ss]))
            print_pf(rslt_test, rslt_train)

        """PFs evaluation"""
        # cl pf: rmse
        train_y_tru, train_y_obv, CLs_pre = rslt_train[:, 2], rslt_train[:, 3], rslt_train[:, 4]
        CLs_tru = comp_cl_upper(train_y_tru, train_y_obv)
        cl_rmse_this = eval_cl(CLs_tru, CLs_pre, False)

        # ctn PFs throughout test steps
        test_y_tru, test_y_pre = rslt_test[:, 1], rslt_test[:, 2]
        pfs_tt_dict = eval_pfs(test_y_tru, test_y_pre)
        gmean_tt, mcc_tt = pfs_tt_dict["gmean_tt"], pfs_tt_dict["mcc_tt"]
        r1_tt, r0_tt = pfs_tt_dict["recall1_tt"], pfs_tt_dict["recall0_tt"]
        prec_tt, f1_tt = pfs_tt_dict["precision_tt"], pfs_tt_dict["f1_score_tt"]

        # assign
        if ss == 0:  # init
            n_row, n_col = gmean_tt.shape[0], len(seed_lst)
            cl_rmse_ss, gmean_tt_ss = np.empty(n_col), np.empty((n_row, n_col))
            r1_tt_ss, r0_tt_ss = np.copy(gmean_tt_ss), np.copy(gmean_tt_ss)
            mcc_tt_ss, prec_tt_ss, f1_tt_ss = np.copy(gmean_tt_ss), np.copy(gmean_tt_ss), np.copy(gmean_tt_ss)
        cl_rmse_ss[ss] = cl_rmse_this
        gmean_tt_ss[:, ss], r1_tt_ss[:, ss], r0_tt_ss[:, ss] = gmean_tt, r1_tt, r0_tt
        mcc_tt_ss[:, ss], prec_tt_ss[:, ss], f1_tt_ss[:, ss] = mcc_tt, prec_tt, f1_tt

    # return for cache ctn pfs across all seeds
    info_pfs = "pfs of all seeds in numpy of the shape (n_test, nb_seed)."
    pfs_ss_dct = {'gmean': gmean_tt_ss, 'mcc': mcc_tt_ss, 'r1': r1_tt_ss, 'r0': r0_tt_ss,
                  'f1': f1_tt_ss, 'prec': prec_tt_ss, 'cl_rmse': cl_rmse_ss,
                  'info': info_pfs}
    info_nb_train = "the number of training sample of all seeds in numpy of shape (nb_seed, )."
    nb_train_dct = {'nb_train_delay': nb_train_delay_np, 'nb_train_human': nb_train_human_np,
                    'acc_train_churn': acc_train_churn_np, 'info': info_nb_train}
    retn_main = {'pfs_ss': pfs_ss_dct, 'nb_train_ss': nb_train_dct}
    if verbose_int >= 0:
        print("%s \n\t>>> Done %s() on %s" % (info_run, a_sdp_runs.__name__, datetime.today()))
    return retn_main


def is_waiting_method(human_dict):
    # judge if it is the waiting-time method.
    # debug on 2024/4/16, debug again 2024-05-28
    if 'has_human'.lower() in human_dict and (not human_dict['has_human']):  # the waiting-time method
        return True
    else:
        return False


def is_human_dict_RQ1(human_dict):
    if 'human_eff'.lower() in human_dict and 'human_err' in human_dict:  # RQ1: HumLa or Eco-HumLa without human VL
        return True
    else:
        return False


def is_humla_study_RQ23(human_dict):
    if 'study_vl_dict'.lower() in human_dict and 'study_least_conf_dict' in human_dict:  # RQ2 and RQ3
        return True
    else:
        return False


def para_classifier(cluster_trained, clf_name, X_ptrn, y_ptrn):
    """2022/8/8 based on TengCong's help.
    tuning paras:
        oza:    (n_tree)
        oob:    (n_tree, theta_imb)
        our:    (n_tree, theta_imb, theta_cl)

    Liyan on 8/19: correct the potential bug for which we forgot to pass DenStream within this func.
    The potential issue would be that label confidence derivation related para-s would not be properly tuned
    as we compute the perfect CLs in this version.
    2024/4/16 update codes for speed-up
    """

    # para tuning
    n_tree_lst = [5, 10, 20, 30, 40]  # 5
    theta_imb_lst = [0.9, 0.95, 0.99, 0.999]  # 4
    theta_cl_lst = [0.8, 0.9]  # 2
    # define evaluation
    seed_auto_tune = 255126  # random_status for repetition

    # define model and its search space
    if clf_name == "oza":
        # define search space
        space = dict()
        space['n_tree'] = n_tree_lst

        # define model
        class our_classifier(BaseEstimator, ClassifierMixin):
            def __init__(self, n_tree=None):
                self.classifier = None
                self.n_tree = n_tree

            def fit(self, X_ptrn, y_ptrn):
                self.classifier = OzaBaggingClassifier(HoeffdingTreeClassifier(), self.n_tree, seed_auto_tune)
                self.classifier.partial_fit(X_ptrn, y_ptrn, label_val)

            def predict(self, testX):
                y_pred = self.classifier.predict(testX)
                return y_pred
    elif clf_name == "oob":
        # define search space
        space = dict()
        space['n_tree'] = n_tree_lst
        space['theta_imb'] = theta_imb_lst

        # define models
        class our_classifier(BaseEstimator, ClassifierMixin):
            def __init__(self, n_tree=None, theta_imb=None):
                self.classifier = None
                self.n_tree = n_tree
                self.theta_imb = theta_imb

            def fit(self, X_ptrn, y_ptrn):
                self.classifier = OzaBaggingClassifier_OOB(
                    HoeffdingTreeClassifier(), self.n_tree, seed_auto_tune, self.theta_imb)
                self.classifier.partial_fit(X_ptrn, y_ptrn, label_val)

            def predict(self, testX):
                y_pred = self.classifier.predict(testX)
                return y_pred
    elif clf_name == "our":
        # define search space
        space = dict()
        space['n_tree'] = n_tree_lst
        space['theta_imb'] = theta_imb_lst
        space['theta_cl'] = theta_cl_lst

        # define models
        class our_classifier(BaseEstimator, ClassifierMixin):
            def __init__(self, n_tree=None, theta_imb=None, theta_cl=None):
                self.classifier = None
                self.n_tree = n_tree
                self.theta_imb = theta_imb
                self.theta_cl = theta_cl

            def fit(self, X_ptrn, y_ptrn):
                self.classifier = OzaBaggingClassifier_OOC(
                    HoeffdingTreeClassifier(), self.n_tree, seed_auto_tune, self.theta_imb, self.theta_cl)
                cl_ptrn = cluster_trained.compute_CLs(X_ptrn, y_ptrn)
                self.classifier.partial_fit(X_ptrn, y_ptrn, cl_ptrn[0], label_val)

            def predict(self, testX):
                y_pred = self.classifier.predict(testX)
                return y_pred
    else:
        raise Exception("Undefined clf_name=%s" % clf_name)
    model = our_classifier()

    # define pf metric
    def gmean_scorer(estimator, X, y_tru):
        estimator.fit(X, y_tru)
        labels = estimator.predict(X)
        r1 = recall_score(y_tru, labels, pos_label=1)
        r2 = recall_score(y_tru, labels, pos_label=0)
        gmean = (r1 * r2) ** (1 / 2)
        return gmean

    # Get the best model and its parameter
    case_use = 2
    if case_use == 0:  # initial strategy in 2023, too time-consuming
        n_split = 3
        nb_repeat = 10
        cv = RepeatedKFold(n_splits=n_split, n_repeats=nb_repeat, random_state=seed_auto_tune)
        grid_search = GridSearchCV(
            model, space, scoring=gmean_scorer, cv=cv, n_jobs=-1, verbose=True)  # verbose=False: no print
        import time
        start_time = time.time()
        result = grid_search.fit(X_ptrn, y_ptrn)  # execute search: taking time
        elapsed_time = time.time() - start_time
        print('elapsed time of cv:', elapsed_time)
        # 2024-4-16
        # Fitting 30 folds for each of 40 candidates, totalling 1200 fits
        # elapsed time of cv: 847.7031359672546
    elif case_use == 1:  # April 2024 new, random search
        para_grid = {
            'n_tree': n_tree_lst,
            'theta_imb': theta_imb_lst,
            'theta_cl': theta_cl_lst
        }
        # Perform randomized search
        random_search = RandomizedSearchCV(
            model, param_distributions=para_grid, scoring=gmean_scorer, cv=5, n_iter=10, verbose=True)
        result = random_search.fit(X_ptrn, y_ptrn)  # execute search: taking time
        # 2024-4-16
        # Fitting 5 folds for each of 10 candidates, totalling 50 fits
        # elapsed time: 48.82977533340454
    elif case_use == 2:  # April 2024 new, grid search; use this
        para_grid = {
            'n_tree': n_tree_lst,
            'theta_imb': theta_imb_lst,
            'theta_cl': theta_cl_lst
        }
        grid_search = GridSearchCV(model, param_grid=para_grid, cv=5, verbose=True)
        result = grid_search.fit(X_ptrn, y_ptrn)
        # 2024-4-16
        # Fitting 5 folds for each of 24 candidates, totalling 120 fits
        # elapsed time - grid_search: 142.75933599472046

    # assign the opt para
    if clf_name == "oza":
        n_tree = result.best_params_['n_tree']
        theta_imb, theta_cl = invalid_val, invalid_val
    elif clf_name == "oob":
        n_tree = result.best_params_['n_tree']
        theta_imb = result.best_params_['theta_imb']
        theta_cl = invalid_val
    elif clf_name == "our":
        n_tree = result.best_params_['n_tree']
        theta_imb = result.best_params_['theta_imb']
        theta_cl = result.best_params_['theta_cl']
    elif clf_name == "pbsa":
        pass
    print("\t\t optimal para: n_tree=%d, theta_imb=%.4f, theta_cl=%.4f"
          % (n_tree, theta_imb, theta_cl))
    return n_tree, theta_imb, theta_cl


def para_denStream(X_norm, y_true, nb_repeat=10):
    """
    Zixin helped me with this task.
    2022/8/9
        We still suffer from some invalid para setups and see warnings.
        But I think it has been enough to
    2024-4-16 update the code to speed up
    """

    # define evaluation
    seed_auto_tune = 2380546
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=nb_repeat, random_state=seed_auto_tune)

    # define search space
    space = dict()
    space['eps'] = loguniform(0.1, 20)
    space['mu'] = loguniform(0.1, 15)
    space['beta'] = uniform(0.2, 0.6)
    space['lambd'] = loguniform(0.125, 1.5)

    # define model
    class our_cluster(BaseEstimator, ClusterMixin):
        def __init__(self, lambd=None, eps=None, beta=None, mu=None):
            self.cluster = None
            self.eps = eps
            self.lambd = lambd
            self.beta = beta
            self.mu = mu

        # y: ignored
        def fit(self, X_norm, y=y_true):
            theta_cl = 0.8  # by default, no impact

            # Zixin modified, 2022/08/09, raise an exception when beta * mu <= 1
            if self.beta * self.mu <= 1:
                raise AssertionError("[Parameters Error]beta * mu <= 1, skipped\n")
            self.cluster = DenStream(theta_cl=theta_cl, lambd=self.lambd, eps=self.eps, beta=self.beta, mu=self.mu)
            self.cluster.partial_fit(X_norm, y)

        def predict(self, testX):
            y_pred = self.cluster.predict(testX)
            return y_pred

    model = our_cluster()

    # define search, y ignored
    def silhouette_scorer(estimator, X, y=y_true):
        try:
            estimator.fit(X)
        except AssertionError as ae:
            print(ae)
            return -1

        labels = estimator.predict(X)
        try:
            score = silhouette_score(X, labels, random_state=seed_auto_tune)
        except ValueError:
            print("[Value Error] All labels are the same: %d" % np.unique(labels)[0])
            return -1
        return score

    search = RandomizedSearchCV(
        model, space, n_iter=200, scoring=silhouette_scorer,
        n_jobs=5, cv=cv, random_state=seed_auto_tune, verbose=False)

    # execute search
    result = search.fit(X_norm, y_true)

    # prepare returns
    eps_opt = result.best_params_['eps']
    mu_opt = result.best_params_['mu']
    beta_opt = result.best_params_['beta']
    lambd_opt = result.best_params_['lambd']
    return eps_opt, mu_opt, beta_opt, lambd_opt


def comp_cl_upper(y_true, y_obv):
    """compute CLs for the upper bound, also the benchmark CLs.
    """
    assert y_true.shape == y_obv.shape, "the shape of y_obv should equal to that of y_true"
    upper_conf_levels = np.ones(np.size(y_true))
    upper_conf_levels[np.where(y_obv != y_true)] = 0
    # upper_conf_levels = np.where(y_true == y_true, 1, 0)
    return upper_conf_levels


class my_norm_scaler:
    """set my normaliser for DenStream in JIT-SDP.
    Note that the 1st fea "fix_bug" does not get involved & we should have 12 fea-s.

    2021-12-4   create
    2022-1-18   move back into main_test_cl_DenStream()
    2022-7-28   update
    """

    def __init__(self, n_fea, norm_name="z_score"):
        self.n_fea = n_fea
        self.norm_name = norm_name  # by default z-score
        if self.norm_name.lower() == "min_max".lower():
            self.norm_scaler = preprocessing.MinMaxScaler()
        elif self.norm_name.lower() == "z_score".lower():
            self.norm_scaler = preprocessing.StandardScaler()

    def check_feature(self, XX):
        assert XX.shape[1] == self.n_fea, "wrong fea number. It should be 13 for transformed jit-sdp."

    def my_fit(self, XX):
        self.check_feature(XX)
        """see comments in my_transform() below"""
        if self.n_fea == 12:  # for jit-sdp: the 1st fea "fix_bug" should NOT be normalised.
            my_norm = self.norm_scaler.fit(XX[:, 1:])
        else:  # for synthetic
            my_norm = self.norm_scaler.fit(XX)
        return my_norm

    def my_transform(self, xx):
        if xx.ndim == 1:  # if xx contains only 1 data_stream sample
            xx = xx.reshape((-1, self.n_fea))
        """the real jit-sdp vs synthetic. 
        This is roughly decided based on # fea: for jit-sdp: n_fea=12; for syn: probably NOT 13.
        """
        if self.n_fea == 12:  # for jit-sdp: the 1st fea "fix_bug" should remain unchanged.
            xx_trans = np.hstack((xx[:, 0].reshape(-1, 1), self.norm_scaler.transform(xx[:, 1:])))
        else:  # for synthetic
            xx_trans = self.norm_scaler.transform(xx)
        return xx_trans


def get_human_auto_name():
    # the string name representing that human effort is automatically determined, such as in HumLa
    return "auto".lower()


def get_human_random_name():
    # for RQ2 and RQ3: the string name representing that human effort and/or noise is randomly chosen.
    return "random".lower()


class HumanDictAnalyzer:
    """Input human_dict related checking and reset
    Liyan updated the original function to the class on 2023/11/14
    """

    def __init__(self, human_dict=None):
        if human_dict is None:  # the default setup
            human_dict = {
                "has_human": False, "human_err": None, "human_eff": None
            }
            warnings.warn("'human_dict' is not set. I adopt the default waiting-time method.")
        self.human_dict = human_dict
        self.has_human = self.human_dict['has_human']
        if not is_waiting_method(self.human_dict):
            self.human_eff = self.human_dict['human_eff']
            self.human_err = self.human_dict['human_err']
        # Call the set_human_dict() method during initialization
        self.set_human_dict()

    def set_human_dict(self):
        self.check_has_human()  # abort if fail
        self.reset_has_human()  # for the waiting time method
        if self.has_human:  # for HumLa and Eco-HumLa
            # human_noise
            self.check_human_noise()
            self.debug_human_noise()
            # human_effort
            self.check_human_effort()
            self.debug_human_effort()

    def is_humla(self):
        if self.has_human and isinstance(self.human_eff, (int, float)):
            return True
        else:
            return False

    def is_eco_humala(self):
        if self.has_human and isinstance(self.human_eff, str):
            self.check_auto_human_effort_name()
            return True
        else:
            return False

    def check_has_human(self):
        assert isinstance(self.has_human, bool), (
                "Error in type(has_human):%s. It should be Bool." % type(self.has_human))

    def reset_has_human(self):
        # update has_human
        if not self.has_human:  # the waiting time method
            self.human_err, self.human_eff = None, None  # reset

    def check_human_noise(self):
        assert isinstance(self.human_err, (int, float)) and 0 <= self.human_err <= 1, \
            "Error in human_err setup."

    def debug_human_noise(self):
        # debug human_noise: if human_noise = 0.0 or 1.0, dir-search cannot find int(human_err)
        if self.human_err == 0 or self.human_err == 1:  # debug human_noise
            self.human_err = int(self.human_err)

    def check_human_effort(self):
        if self.is_humla():
            assert 0 < self.human_eff <= 1, "Error in the value range of human_eff=%.4f" % self.human_eff
        elif self.is_eco_humala():
            self.check_auto_human_effort_name()
        else:
            raise Exception("Error in type(human_eff)=%s. Check the human_dict." % type(self.human_eff))

    def debug_human_effort(self):
        # debug human_effort: if human_effort = 0.0 or 1.0, dir-search cannot find int(human_eff)
        if self.human_eff == 0 or self.human_eff == 1:
            self.human_eff = int(self.human_eff)

    def check_auto_human_effort_name(self):
        assert self.human_eff.lower() == get_human_auto_name(), (
                "Error in human_eff setup. PLS use '%s'." % get_human_auto_name())


def rslt_dir_RQ1(clf_name, human_dict, project_id, wait_days, n_trees, theta_imb, theta_cl):
    """
    Set the result directory that store the training and test result for each seed.
    2022-7-30, 2022/10/13, 2022/12/23, 2023/11/14
    """
    clf_name = clf_name.lower()
    pre_to_dir = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name
    # get human effort and noise
    human_analyzer = HumanDictAnalyzer(human_dict)
    if human_analyzer.has_human:
        pre_to_dir += "-human/effort"
        if human_analyzer.is_humla():
            pre_to_dir += str(human_analyzer.human_eff)
        elif human_analyzer.is_eco_humala():
            pre_to_dir += "-%s" % human_analyzer.human_eff
        pre_to_dir += "/error" + str(human_analyzer.human_err) + "/" + str(wait_days) + "d"  # note to use str()
    else:
        pre_to_dir += "/" + str(wait_days) + "d"  # note to use str()
    # para-s for each classifier
    to_dir = pre_to_dir + "/n_trees" + str(n_trees)
    if clf_name != "oza":
        to_dir += "-theta_imb" + str(theta_imb)
    if clf_name != "oza" and clf_name != "oob" and clf_name != "oob_filter":
        to_dir += "-theta_cl" + str(theta_cl)
    return to_dir


class HumLaStudyAnalyzer:
    """Analyzer for RQ2 and RQ3:
    RQ2: To what extent predictive pf would be impacted when human labeling is not immediate due to VL?
    RQ3: To what extent it is useful to include some least-confident defect-predictions for learning purposes?
        How many least-confident defect predictions should be included in order to improve predictive PF of Eco-HumLa?
    Liyan Song on 2023/11/28
    """

    def __init__(self, humla_study_dict=None):
        if humla_study_dict is None:
            humla_study_dict = {
                'study_vl_dict': {'is_study_vl': True, 'which_method': 'eco', 'vl_hour': 2},  # for RQ2
                'study_least_conf_dict': {'is_study_least_conf': False, 'prob_least_conf': None}  # for RQ3
            }
            warnings.warn("'humla_study_dict' for RQ2 or RQ3 is not set. The default is adopted")
        self.humla_study_dict = humla_study_dict
        self.check_input_set()  # abort if failed
        # the same member variables of HumLa
        self.has_human = True
        self.human_err = get_human_random_name()
        if self.is_study_RQ2_vl_humla():
            self.human_eff = get_human_random_name()
        elif self.is_study_RQ2_vl_eco():
            self.human_eff = get_human_auto_name()
        elif self.is_study_RQ3_eco_least():
            self.human_eff = get_human_auto_name()
        else:
            raise Exception("Undefined case for human effort")

    # overall check: in __init__()
    def check_input_set(self):
        self.check_RQ2_bool_study_vl()
        self.check_RQ2_humla_opts()
        self.check_RQ2_vl()
        self.check_RQ3_bool_eco_least()
        self.check_RQ3_study_eco_prob_least()
        # check potential conflicting settings
        if self.is_study_RQ2_vl() == self.is_study_RQ3_eco_least():
            raise Exception("Conflict in investigating either both RQ2 and RQ3 or none of them.")
        if self.is_study_RQ2_vl():  # for RQ2
            if self.is_study_RQ2_vl_eco() == self.is_study_RQ2_vl_humla():
                raise Exception("Conflicts in investigating VL for either both humla and eco-humla or none of them.")

    # for RQ2: the below function members
    def check_RQ2_bool_study_vl(self):
        assert isinstance(self.humla_study_dict['study_vl_dict']['is_study_vl'], bool)

    def is_study_RQ2_vl(self):
        if self.humla_study_dict['study_vl_dict']['is_study_vl']:
            return True
        else:
            return False

    def get_reset_RQ2_humla_opt(self):
        humla_option = self.humla_study_dict['study_vl_dict']['which_method']
        if humla_option is not None:
            humla_option = humla_option.lower()
        if self.is_study_RQ2_vl():
            return humla_option
        else:
            return None

    def check_RQ2_humla_opts(self):
        assert self.get_reset_RQ2_humla_opt() in ('eco', 'eco-humla', 'eco_humla', 'humla', None)

    def is_study_RQ2_vl_eco(self):
        self.check_RQ2_humla_opts()  # abort if failed
        if self.is_study_RQ2_vl() and self.get_reset_RQ2_humla_opt() in ('eco', 'eco_humla', 'eco-humla'):
            return True
        else:
            return False

    def is_study_RQ2_vl_humla(self):
        self.check_RQ2_humla_opts()  # abort if failed
        if self.is_study_RQ2_vl() and self.get_reset_RQ2_humla_opt() == 'humla':
            return True
        else:
            return False

    def get_RQ2_vl_hour(self):
        if self.is_study_RQ2_vl():
            return self.humla_study_dict['study_vl_dict']['vl_hour']
        else:
            return 0  # 2024-05-30 debug: for RQ3 and RQ1, human-vl is assigned zero

    def check_RQ2_vl(self):
        if self.is_study_RQ2_vl():
            assert isinstance(self.get_RQ2_vl_hour(), (int, float))

    # for RQ3: the below function members
    def check_RQ3_bool_eco_least(self):
        assert isinstance(self.humla_study_dict['study_least_conf_dict']['is_study_least_conf'], bool)

    def is_study_RQ3_eco_least(self):
        if self.humla_study_dict['study_least_conf_dict']['is_study_least_conf']:
            return True
        else:
            return False

    def get_RQ3_eco_prob_least(self):
        if self.is_study_RQ3_eco_least():
            return self.humla_study_dict['study_least_conf_dict']['prob_least_conf']
        else:
            return None

    def check_RQ3_study_eco_prob_least(self):
        if self.is_study_RQ3_eco_least():
            assert isinstance(self.get_RQ3_eco_prob_least(), (int, float))


def rslt_dir_RQ23(clf_name, humla_study_dict, project_id, wait_days, n_tree, theta_imb, theta_cl):
    """for RQ2 and RQ3:
    Set the result directory that store the training and test result for each seed.
    Liyan Song on 2023/11/28 study eco-humla to investigate the most vs least confident defect-predictions
    """
    clf_name = clf_name.lower()
    # para-s for each classifier
    dir_para = "/n_trees" + str(n_tree)
    if clf_name != "oza":
        dir_para += "-theta_imb" + str(theta_imb)
    if clf_name != "oza" and clf_name != "oob" and clf_name != "oob_filter":
        dir_para += "-theta_cl" + str(theta_cl)
    # dir setting
    common_dir = "%s/%s/%sd" % (data_id_2name(project_id), clf_name, str(wait_days))
    humla_study_analyzer = HumLaStudyAnalyzer(humla_study_dict)
    if humla_study_analyzer.is_study_RQ2_vl():  # RQ2
        vl_days = humla_study_analyzer.get_RQ2_vl_hour()
        pre_dir = "%sRQ2-vl/%s/%s/vl_%sh" % (
            dir_rslt_save, humla_study_analyzer.get_reset_RQ2_humla_opt(), common_dir, str(vl_days))
    elif humla_study_analyzer.is_study_RQ3_eco_least():  # RQ3
        prob_least_conf = humla_study_analyzer.get_RQ3_eco_prob_least()
        pre_dir = "%sRQ3-eco/%s/prob_least%s" % (dir_rslt_save, common_dir, str(prob_least_conf))
    else:
        raise Exception("Error in opting for one of the two RQs")
    to_dir = pre_dir + dir_para
    return to_dir


def rslt_dir_analyze(to_dir, clf_name, nb_test, seed):
    """
    A method used in jit_sdp_1call().
    Analyse filenames in the directory 'to_dir' and find 'T' that is larger than nb_data,
    so that we can save computational cost to downside load it
    Liyan Song Last updated on 2022/8/18
    2023/11/28 consider the study of eco-humla that investigates the most vs least confident defect predictions
    """
    exist_result, nb_test_saved = False, ""
    fold_names = next(os.walk(to_dir))[1]
    if len(fold_names) > 0:
        for _, fold_name in enumerate(fold_names):
            nb_test_saved = int(fold_name[fold_name.find("T") + 1:])
            if nb_test_saved >= nb_test:
                to_dir_4save = to_dir
                to_dir += "/T" + str(nb_test_saved) + "/"
                flnm_test = to_dir + clf_name + ".rslt_test.s" + str(seed)
                flnm_train = to_dir + clf_name + ".rslt_train.s" + str(seed)
                exist_result = os.path.exists(flnm_test) and os.path.exists(flnm_train)
                if exist_result:
                    break
                else:
                    """handle empty (e.g.) T5000 folder"""
                    to_dir = to_dir_4save
    return exist_result, to_dir, nb_test_saved


def plot_on_1pf(pf_tt, clf_lst, title_info, save_plot=False, to_dir_png=None, my_ylim=None):
    """Plots of online pf-s of multiple classifiers in a single plot.
    parameters:
        pf_tt: shape (nb_test, nb_classifier)
    2022-8-1
    """
    if not (isinstance(clf_lst, list) or isinstance(clf_lst, tuple)):
        raise Exception("Error: clf_lst should be a list")
    if np.ndim(pf_tt) == 1:
        pf_tt = pf_tt[:, np.newaxis]
    if pf_tt.shape[1] != len(clf_lst):
        raise Exception("Error: # classifier NOT matches column size of pf_tt")
    # plot
    xx = np.array(range(pf_tt.shape[0]))  # shape (nb_test,)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for cc, clf_name in enumerate(clf_lst):
        ax.plot(xx, pf_tt[:, cc], label=clf_name)
    # setup
    ax.set_title(title_info)
    if my_ylim is None:
        plt.ylim((0, 1))
    else:
        plt.ylim(my_ylim)
    # ax.grid(True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.legend(loc="best")
    # show/plot
    if not save_plot:
        plt.show()
    else:
        if to_dir_png is None:  # set default to_dir_png
            to_dir_png = dir_plot_root + "tmp_pfs_throughout_time/"
        os.makedirs(to_dir_png, exist_ok=True)
        to_flnm = to_dir_png + title_info
        plt.savefig(to_flnm + ".png")
        print("\tPF plot is saved in %s" % to_dir_png)


def print_pf(rslt_test, rslt_train):
    """
    rslt_test ~ (test_time, y_true, y_pred)
    rslt_train ~ (commit_time, use_time, yy, y_obv, cl, use_cluster)
    2022-8-2
    """
    # extract data_stream info
    test_y_tru, test_y_pre = rslt_test[:, 1], rslt_test[:, 2]
    y_train_tru_all, y_train_obv_all, cl_pre = rslt_train[:, 2], rslt_train[:, 3], rslt_train[:, 4]

    # training label noise
    nb_train, nb_train_noise = y_train_obv_all.shape[0], len(np.where(y_train_tru_all != y_train_obv_all)[0])
    trn_label_noise = nb_train_noise / nb_train  # vip
    print("\t training: label_noise=%f" % trn_label_noise)

    # training 1-sided label noise
    nb_defect = np.sum(y_train_obv_all == 1)
    trn_1side_noise = nb_train_noise / nb_defect
    print("\t training: 1sided_noise=%f" % trn_1side_noise)

    # c1% of test data_stream, i.e., true c1%
    nb_test_c1, nb_test = len(np.where(test_y_tru == 1)[0]), test_y_tru.shape[0]
    tst_c1_percent = nb_test_c1 / nb_test
    print("\t testing: class imbalance c1%%=%f" % tst_c1_percent)

    # pf: rmse
    cl_tru = comp_cl_upper(y_train_tru_all, y_train_obv_all)
    eval_cl(cl_tru, cl_pre, True)
    # pf: online prediction
    eval_pfs(test_y_tru, test_y_pre, True)


def eval_cl(CL_tru, CL_est, verbose=False):
    """evaluate estimated label confidence
    2022-8-1    separate from uti_eval_pfs
    """
    rmse = mean_squared_error(CL_tru, CL_est, squared=False)
    if verbose:
        print("\t rmse of cl_est=%f." % rmse)
    return rmse


def eval_pfs(test_y_tru, test_y_pre, verbose=False):
    """evaluate PFs in terms of g-mean, recall-1, recall-0..
    2022/6/2        Separate this func.
    2022/11/28      insert mcc and other pf metrics
    """
    # ave PFs across test steps
    theta_eval = 0.99
    pfs_tt_dct = compute_online_PF(test_y_tru, test_y_pre, theta_eval)
    gmean_ave_tt = np.nanmean(pfs_tt_dct["gmean_tt"])
    r1_ave_tt, r0_ave_tt = np.nanmean(pfs_tt_dct["recall1_tt"]), np.nanmean(pfs_tt_dct["recall0_tt"])
    if verbose:
        print("\t ave online gmean=%.4f, r1=%.4f, r0=%.4f" % (gmean_ave_tt, r1_ave_tt, r0_ave_tt))
    return pfs_tt_dct


def set_para_enu(clf_name, n_tree_lst, theta_imb_lst, theta_cl_lst):
    """set the enumerated para settings
    As methods such as oob vs our will have different para settings,
    the possible enumerated para settings would be different accordingly.
    This method is used to get the enumerated para settings for sdp methods individually.

    2022-1-18   extract to form this code from get_para_bst()
    2022-6-2    adapt this func.
    """
    clf_name = clf_name.lower()

    # para_enu~(n_tree, theta_imb, theta_cl)
    if clf_name == "oza":
        para_enu = product(n_tree_lst, [invalid_val], [invalid_val], [invalid_val])
    elif clf_name == "oob" or clf_name == "oob_filter":
        para_enu = product(n_tree_lst, theta_imb_lst, [invalid_val], [invalid_val])
    elif clf_name == "our":
        para_enu = product(n_tree_lst, theta_imb_lst, theta_cl_lst, [invalid_val])
    else:
        raise Exception("Undefined classifier with clf_name=%s." % clf_name)

    # usage of para_enu
    # for pp, para in enumerate(para_enu):
    #     n_tree, theta_imb, theta_cl = para[0], para[1], para[2]
    # para_enu~(n_tree, theta_imb, theta_cl)
    return para_enu


if __name__ == "__main__":
    # Parameter Human setup for RQ1-RQ3:
    # human_dict_RQ1_humla = {
    #     "has_human": True, "human_err": 0.1, "human_eff": "auto"  # for RQ1
    # }
    # human_dict_RQ1_eco = {
    #     "has_human": True, "human_err": 0.1, "human_eff": 0.8  # for RQ1
    # }
    # human_dict_RQ2_humla = {
    #     'study_vl_dict': {'is_study_vl': True, 'which_method': 'humla', 'vl_hour': 2},  # for RQ2
    #     'study_least_conf_dict': {'is_study_least_conf': False, 'prob_least_conf': None}  # for RQ3
    # }
    # human_dict_RQ2_eco = {
    #     'study_vl_dict': {'is_study_vl': True, 'which_method': 'eco', 'vl_hour': 2},  # for RQ2
    #     'study_least_conf_dict': {'is_study_least_conf': False, 'prob_least_conf': None}  # for RQ3
    # }
    # human_dict_RQ3 = {
    #     'study_vl_dict': {'is_study_vl': False, 'which_method': None},  # for RQ2
    #     'study_least_conf_dict': {'is_study_least_conf': True, 'prob_least_conf': 0.2}  # for RQ3
    # }
    # # run jit-sdp with HumLa
    # human_dict_RQ2_humla = {
    #     'study_vl_dict': {'is_study_vl': True, 'which_method': 'humla', 'vl_hour': 2},  # for RQ2
    #     'study_least_conf_dict': {'is_study_least_conf': False, 'prob_least_conf': None}  # for RQ3
    # }
    # a_sdp_runs(project_id=9, human_dict=human_dict_RQ2_humla, seed_lst=[20], just_run=False)
    # further investigate RQ2
    # RQ2_analyze_data_def_vl()
    # RQ2_analyze_test_time()

    # 2024-07-25 again to re-check RQ2-human-vl
    # common para
    project_id = 0  # 9-rails pf(human delay) the largest impact
    seed_lst = [0]
    just_run = False
    just_load = False
    humla_opt = "humla"
    # dif para settings
    # for local PC
    # human_dict1 = {
    #     'study_vl_dict': {'is_study_vl': True, 'which_method': humla_opt, 'vl_hour': 0},  # for RQ2
    #     'study_least_conf_dict': {'is_study_least_conf': False, 'prob_least_conf': None}  # for RQ3
    # }
    # # for 319pc
    # human_dict2 = {
    #     'study_vl_dict': {'is_study_vl': True, 'which_method': humla_opt, 'vl_hour': 2},  # for RQ2
    #     'study_least_conf_dict': {'is_study_least_conf': False, 'prob_least_conf': None}  # for RQ3
    # }
    # human_dict = {
    #     "has_human": False, "human_err": None, "human_eff": None  # for RQ1
    # }
    # a_sdp_runs(human_dict=human_dict, project_id=project_id, seed_lst=seed_lst, just_run=just_run, just_load=just_load)

    # 2024/3/21 restart the analyzing
    # RQ2_analyze_plot_PF()  # tmp 2024/3/21
    # RQ2_analyze_plot_all()

    # example_sdp_clf(example_id=0, project_id=0, just_run=False)
    auto_para_clf(project_id=100)
