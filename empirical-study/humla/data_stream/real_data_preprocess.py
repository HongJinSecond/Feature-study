import numpy as np


def real_data_preprocess(X_org):
    """ feature pre-process: [2021-11-23], updated on 2022-11-2
    The original feature number is 14, and the converted feature number is reduced to 12.

    We pre-process the features according to Kamei's TSE2013.
    The implementation is from "2019 Local vs global models for jit-sdp - online clustering".
    But later in 2022-11, we found their codes with bugs and Liyan updated this func.
    Please see Liyan's report and NB-pg.95 for more details.

    Input & output arguments
        X_org: (n_sample, n_fea=14)
    return: preprocessed data_stream features
    """
    _, n_fea = X_org.shape
    assert n_fea == 14, "wrong dim of jit-sdp X_org"
    # manual setup carefully
    id_fix, id_ns, id_nd, id_nf, id_entropy = 0, 1, 2, 3, 4
    id_la, id_ld, id_lt, id_ndev, id_age = 5, 6, 7, 8, 9
    id_nuc, id_exp, id_rexp, id_sexp = 10, 11, 12, 13

    """Eliminate invalid entries from X_org. 
    The meanings of some features are annotated as below:
        * lt:   lines of code in a file before the change.
        * age:  the average time interval between the last and the current change.
        * rexp: recent developer experience
        * nf:   # modified files.
    Therefore, they should be non-negative in practice. 
    
    Also, negative values may induce technical errors. 
    For instance, in "log2", the potential log2(negative_value) will report two warnings:
        * RuntimeWarning: divide by zero encountered in log2
        * RuntimeWarning: invalid value encountered in log2 
    We may need to consider more features later on when dealing with more jit-sdp datasets.
    
    [2021-11-23] and updated on 2022-11-2.
    """

    # [bug] Code defect found by YiboTang on 2023/11/2
    # see: https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html
    # np.logical_and() has two valid input arguments and thus the third logic term is not executed correctly,
    # meaning, the third output element is invalid.
    # use_data = np.logical_and(X_org[:, id_lt] >= 0,
    #                           X_org[:, id_age] >= 0,
    #                           X_org[:, id_rexp] >= 0)
    # Nevertheless, this defect should not largely affect the conclusions of our previous studies
    # because the impacted data entries (the # of changes that have negative rexp value) were not large.
    use_data = np.logical_and(
        np.logical_and(X_org[:, id_lt] >= 0, X_org[:, id_age] >= 0),
        X_org[:, id_rexp] >= 0
    )
    # Nov.2: Not yet testing the above correcting code

    X_org = np.copy(X_org[use_data, :])
    X_trans = np.copy(X_org)

    """cumulative churn, Kamei's decChurn.r"""
    churn_np = (X_org[:, id_la] + X_org[:, id_ld]) / 2

    """feature pro-process"""
    # 1. deal with multi-collinearity
    # (1.1) LA = LA / LT; LD = LD / LT
    select_lt = X_trans[:, id_lt] >= 1  # avoid zero-denominator when lt==0
    X_trans[select_lt, id_la] = X_trans[select_lt, id_la] / X_trans[select_lt, id_lt]
    X_trans[select_lt, id_ld] = X_trans[select_lt, id_ld] / X_trans[select_lt, id_lt]

    # (1.2) LT = LT / NF; NUC = NUC / NF
    select_nf = X_trans[:, id_nf] >= 1  # avoid zero-denominator when nf=0
    X_trans[select_nf, id_lt] = X_trans[select_nf, id_lt] / X_trans[select_nf, id_nf]
    X_trans[select_nf, id_nuc] = X_trans[select_nf, id_nuc] / X_trans[select_nf, id_nf]

    # (1.3) entropy = entropy / NF   refer TSE'13 Kamei predUtils.r
    # [Kamei-TSE2013 code] if the num of files is less than 2, entropy is not normalized.
    select_nf = X_trans[:, id_nf] >= 2
    X_trans[select_nf, id_entropy] = X_trans[select_nf, id_entropy] / np.log2(X_trans[select_nf, id_nf])

    # (1.4) remove ND and REXP
    X_trans = X_trans[:, np.setdiff1d(range(n_fea), np.array((id_nd, id_rexp)))]

    # 2. logarithmic transformation
    n_fea_new = X_trans.shape[1]
    ids2_ = np.setdiff1d(range(n_fea_new), id_fix)  # Note that id_fix remains unchanged indeed.
    X_trans[:, ids2_] = X_trans[:, ids2_] + 1  # refer Kamei factorMain.r Line 38
    X_trans[:, ids2_] = np.log2(X_trans[:, ids2_])

    # 2022-11-7 churn should be aligned with X_trans
    # return X_trans, use_data, churn_np
    return np.hstack((X_trans, np.array([churn_np]).T)), use_data


# The below is the preprocessing function
#   of the paper "2019 Local vs global models for jit-sdp - online clustering"
#   from https://github.com/yangxingguang/LocalJIT
# However, Liyan found it to be erroneous.
# Refer to Liyan's report-2022-11 or NB-pg95.
#
#
# def preprocessing(X_org):
#     # calculate  churn  FSE'16 utils.R
#     la = X_org[:, 4]
#     ld = X_org[:, 5]
#     nf = X_org[:, 2]
#     lt = X_org[:, 6]
#     lt_ = lt*nf
#     lt[lt == 0] = 1
#     churn = (la + ld)*lt_/2
#
#     # (1)deal with multi-collinearity
#     #  1.1 LA = LA / LT; LD = LD / LT
#     select_lt = X_org[:, 6] >= 1
#     X_org[select_lt, 4] = X_org[select_lt, 4] / X_org[select_lt, 6]
#     X_org[select_lt, 5] = X_org[select_lt, 5] / X_org[select_lt, 6]
#     #  1.2 LT = LT / NF; NUC = NUC / NF
#     select_nf = X_org[:, 2] >= 1
#     X_org[select_nf, 6] = X_org[select_nf, 6] / X_org[select_nf, 2]
#     X_org[select_nf, 10] = X_org[select_nf, 10] / X_org[select_nf, 2]
#     # 1.3 entropy = entropy / NF   refer TSE'13 Kamei predUtils.r
#     select_nf = X_org[:, 2] >= 2
#     X_org[select_nf, 3] = X_org[select_nf, 3] / np.log2(X_org[select_nf, 2])
#     # 1.4 remove ND and REXP
#     X_org = X_org[:, (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13)]
#     # (2)logarithmic transformation
#     X_org[:, (2, 3, 4, 5, 7, 8, 9, 10, 11)] = X_org[:, (2, 3, 4, 5, 7, 8, 9, 10, 11)] + 1
#     X_org[:, (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11)] = np.log2(X_org[:, (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11)])
#     X_org = X_org[:, (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 3, 4)]
#     return np.hstack((X_org, np.array([churn]).T))
#
# The following function demonstrates how to utilize the preprocessing function.
# def read_csv_grouped_month(path):
#     data_months = []
#     data_stream = pd.read_csv(path).values
#     x_data = data_stream[:, 2:-1]
#     x_data = x_data.astype(np.float64)
#     # 数据预处理
#     x_data = preprocessing(x_data)  # Liyan: all preprocessed fea-s including churn are used.
#     y_data = data_stream[:, -1]  # Liyan: the label is assigned.
#     y_data = y_data.astype(np.bool)
#     commitdate = data_stream[:, 1]
#     current_month = str_2_month(commitdate[0])
#     current_first = 0
#     for i in range(len(commitdate)):
#         if str_2_month(commitdate[i]) != current_month:
#             x_month = x_data[current_first:i]
#             y_month = y_data[current_first:i]
#             data_months.append([x_month, y_month])
#             current_first = i
#             current_month = str_2_month(commitdate[i])
#     x_month = x_data[current_first:len(commitdate)]
#     y_month = y_data[current_first:len(commitdate)]
#     data_months.append([x_month, y_month])
#     return data_months
