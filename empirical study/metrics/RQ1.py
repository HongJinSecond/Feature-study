import numpy as np
from scipy import stats
import pandas as pd

def Zero_Norm(x:np.ndarray,y:np.ndarray)->int:
    """

    :param x:one vector
    :param y:another vector
    :return:
    """
    return (x!=y).sum()


def Percentage(x:np.ndarray,y:np.ndarray)->float:
    """

    :param x:
    :param y:
    :return:
    """
    total=(x!=y).sum()
    return total/len(x)

def L_1_Norm(x:np.ndarray,y:np.ndarray)->float:
    """

    :param x:
    :param y:
    :return:
    """
    return (np.abs(x-y).sum())/len(x)

def L_2_Norm(x:np.ndarray,y:np.ndarray)->float:
    """

    :param x:
    :param y:
    :return:
    """
    return (np.square(x-y).sum())/len(x)

def Spearman_correlation(x:np.ndarray,y:np.ndarray)->float:
    """

    :param x:
    :param y:
    :return:
    """
    return stats.spearmanr(x,y)[0]

def Pearson_correlation(x:np.ndarray,y:np.ndarray)->float:
    """

    :param x:
    :param y:
    :return:
    """
    Cov=((x-x.mean())*(y-y.mean())).sum()
    sigma_x=np.sqrt(((x-x.mean())**2).sum())
    sigma_y=np.sqrt(((y-y.mean())**2).sum())
    return Cov/(sigma_x*sigma_y)

def Wilcoxon(x:np.ndarray,y:np.ndarray)->float:
    """

    :param x:
    :param y:
    :return:
    """
    return stats.wilcoxon(x,y).pvalue


def Result_Analyse(x: pd.DataFrame, y: pd.DataFrame, features: list, metrics: list) -> pd.DataFrame:
    """

    :param x: a vector
    :param y: another vector
    :param features: a list of features that u want to input in yout experiment
    :param metrics: a list of metric functions u want to use in this step
    :return:
    """
    data = []
    column_list = [metric.__name__ for metric in metrics]
    for feature in features:
        output = []
        for metric in metrics:
            output.append(metric(x[feature].values, y[feature].values))
        data.append(output)

    return pd.DataFrame(data, index=features, columns=column_list)


