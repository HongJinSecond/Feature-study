import numpy as np
from sklearn.metrics import recall_score, balanced_accuracy_score, f1_score


def eval_clf_offline(result_np):
    """for pp-report
    :param result_np: (time, y_true, y_pred), created in xxx.jit_sdp_1call()
    :return:
    Liyan on 2021-10-12
    """
    # print(info_str) >> the index is 0: time, 1: y_true, 2: y_pred
    actual_labels = result_np[:, 1]
    predict_labels = result_np[:, 2]

    recall0, recall1, gmean, balanced_acc, f1 = evaluate_offline(actual_labels, predict_labels)

    pf_metrics = gmean, recall0, recall1  # pp-report, 3 values
    metric_names = np.array(("gmean", "recall0", "recall1"))
    return pf_metrics, metric_names


def evaluate_offline(actual_labels, predict_labels):
    """
    :param actual_labels: true yy in numpy, either 0 (clean) or 1 (defect-inducing)
    :param predict_labels: predicted yy in numpy, either 0 (clean) or 1 (defect-inducing)
    :return:
    """

    recall_1 = recall_score(actual_labels, predict_labels, pos_label=1, average='binary')
    recall_0 = recall_score(actual_labels, predict_labels, pos_label=0, average='binary')
    gmean = compute_gmean(np.array([recall_0, recall_1]))

    balanced_accuracy = balanced_accuracy_score(actual_labels, predict_labels)
    f1 = f1_score(actual_labels, predict_labels)

    return recall_0, recall_1, gmean, balanced_accuracy, f1


def compute_gmean(recalls):
    gmeans = 1
    for r in recalls:
        gmeans = gmeans * r
    gmeans = pow(gmeans, 1 / len(recalls))

    return gmeans