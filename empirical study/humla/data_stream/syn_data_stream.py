import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skmultiflow.data import ConceptDriftStream
from humla.utility import check_random_state
from collections import Counter
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from skmultiflow.data.sine_generator import SineGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.led_generator import LEDGenerator
from skmultiflow.data.agrawal_generator import AGRAWALGenerator


def gen_syn_data_stream(syn_type="sine1", drift_type="abrupt",
                        n_data=3000, c1_percent=0.2, rnd_seed=0, is_plot=True):
    """
    Generate an imbalanced synthetic X_org stream.
    website for syn generation: https://scikit-multiflow.readthedocs.io/en/stable/api/api.html

    Setup: class 1 is the majority and class 0 is the minority.
    Parameters:
        tst_c1_percent = c1%, be between 0 and 1, and smaller than 1/2 to represent clas imbalance.
        noise_str - 2sided or 1sided

    2022-3-10   Set this func with 1 cd
    2022-5-05   Upgrade to more types of synthetic streams
    """
    my_rng = check_random_state(seed=rnd_seed)
    assert 0 < c1_percent < 1, "This represents the percentage of class 1 over the whole X_org stream"

    # syn X_org setup for each concept
    n_data1 = round(n_data/2)  # for each concept
    n_c1 = round(n_data1 * c1_percent)
    if syn_type.lower() == "sine1".lower():
        # SINE1 with 2 numerical features that vary from 0~1. Shuo TNNLS'18 used as the 1st syn X_org
        # 5/5 NOTE The X_org stream with the 2 concepts, and it results in the same consequence as label noise.
        stream = SineGenerator(classification_function=0,
                               random_state=my_rng, balance_classes=True, has_noise=False)
        stream_drift = SineGenerator(classification_function=1,
                                     random_state=my_rng, balance_classes=True, has_noise=False)
    elif syn_type.lower() == "sine1_tmp".lower():  # tmp
        stream = SineGenerator(classification_function=0, random_state=rnd_seed, balance_classes=True, has_noise=False)
        stream_drift = SineGenerator(classification_function=3, random_state=rnd_seed,
                                     balance_classes=True, has_noise=False)
        raise Exception("2022-5-11 Decide not to include it in the investigation")

    elif syn_type.lower() == "sea".lower():
        # SEA moving hyperplane concepts with 2 numerical features:
        stream = SEAGenerator(classification_function=2,
                              random_state=my_rng, balance_classes=True, noise_percentage=0.0)
        stream_drift = SEAGenerator(classification_function=3,
                                    random_state=my_rng, balance_classes=True, noise_percentage=0.0)

    elif syn_type.lower() == "rotating_hyperplane":
        # rotating hyperplane:
        # We introduce change to this dataset by adding drift to each weight feature
        # w_i = w_i+d*\sigma, # where \sigma is the probability that the direction of change is reversed
        # and d is the change applied to every example.
        stream = HyperplaneGenerator(n_features=2, n_drift_features=2, mag_change=0.0,
                                     noise_percentage=0, sigma_percentage=0, random_state=my_rng)
        stream_drift = HyperplaneGenerator(n_features=2, n_drift_features=2, mag_change=1.0,
                                           noise_percentage=0, sigma_percentage=1.0, random_state=my_rng)

    elif syn_type.lower() == "stagger_boolean".lower():
        # Stagger Boolean concepts with 3 ordinal/categorical features:
        # size (small, medium and large), shape (circle, square and triangle) and color (red, blue and green)
        # @para classification_function: 0~2
        stream = STAGGERGenerator(classification_function=0, random_state=my_rng, balance_classes=True)
        stream_drift = STAGGERGenerator(classification_function=1, random_state=my_rng, balance_classes=True)

    elif syn_type.lower() == "Agrawal".lower():
        # Agrawal has 9 features (6 numerical + 3 categorical), binary class labels.
        # Presumably, these determine whether the loan should be approved.
        # @para classification_function: 0~9
        # @para perturbation: 0.0~1.0, the probability that noise will happen in the generation.
        stream = AGRAWALGenerator(classification_function=0,
                                  random_state=my_rng, balance_classes=True, perturbation=0)
        stream_drift = AGRAWALGenerator(classification_function=1,
                                        random_state=my_rng, balance_classes=True, perturbation=0)
    elif syn_type.lower() == "LED-not-investigate".lower():
        # LED has 7 binary features (17 irrelevant features are absent in our exp),
        # so it is NOT investigated in this exp.
        stream = LEDGenerator(random_state=my_rng, noise_percentage=0, has_noise=False)
        stream_drift = LEDGenerator(random_state=my_rng, noise_percentage=1, has_noise=False)
        raise Exception("LED is NOT investigated in this study for all binary features.")

    else:
        raise Exception("undefined syn_name=%s" % syn_type)
    n_c0 = n_data1 - n_c1
    n_data1_all = n_data1 * 4  # k=4 times should be enough todo if error evoked, increase "k"

    """
    Generate an over-lengthened stream with 1 concept drift of the P(y|X)-type.
    Refer to the file of skmultiflow.X_org.ConceptDriftStream as below:
        The sigmoid function is an elegant and practical solution to define the probability that 
        each new instance of the stream belongs to the new concept after the drift. 
        The sigmoid function introduces a gradual, smooth transition whose duration is controlled 
        with two parameters: 1) p, the position of the change, and 2) w, the width of the transition.
        The sigmoid function at sample t is f(t) = 1/(1+ exp(-4(t-p)/w)).    
    """
    if drift_type.lower() == "abrupt":
        drift_width = 1
    elif drift_type.lower() == "gradual_fast":
        drift_width = round(n_data1_all / 3)
    elif drift_type.lower() == "gradual_slow":
        drift_width = round(n_data1_all * 2 / 3)
    else:
        raise Exception("undefined syn_name=%s" % drift_type)
    drift_central_pos = n_data1_all
    stream = ConceptDriftStream(stream=stream, drift_stream=stream_drift,
                                position=drift_central_pos, width=drift_width, random_state=my_rng)
    # get all X_org
    X_all, y_all = stream.next_sample(n_data1_all*2)
    Data = np.column_stack((np.array(range(len(y_all))).reshape((-1, 1)), X_all, y_all))

    """sampling to get imbalanced X_org streams
    """
    y_all = Data[:, -1]
    id_c0, id_c1 = np.where(y_all == 0)[0], np.where(y_all == 1)[0]
    id_c0_use = np.sort(id_c0[my_rng.choice(len(id_c0), size=n_c0*2, replace=False)])
    id_c1_use = np.sort(id_c1[my_rng.choice(len(id_c1), size=n_c1*2, replace=False)])
    id_use = np.sort(np.concatenate((id_c0_use, id_c1_use)))
    X, y_true = Data[id_use, 1:-1], Data[id_use, -1]

    if is_plot:  # 2D plot
        X_plt = X
        plot_info = "org"
        if X.shape[1] > 2:
            norm_scaler = my_norm_scaler(n_fea=X.shape[1], norm_name="min_max")
            norm_scaler.my_fit(X)
            X_norm = norm_scaler.my_transform(X)
            pca = PCA(n_components=2)
            X_plt = pca.fit_transform(X_norm)
            print("pca.explained_var_ratio are ", pca.explained_variance_ratio_)
            print("pca.singular_values are ", pca.singular_values_)
            plot_info = "pca"  # overwrite

        fig, axs = plt.subplots(2)
        fig.suptitle("%s: c1%%=%.2f, noise-free, %s CD, %s" % (syn_type, c1_percent, drift_type, plot_info))
        axs[0].scatter(X_plt[:n_data1, 0], X_plt[:n_data1, 1], c=y_true[:n_data1])
        axs[0].grid(True)
        axs[0].set_ylabel("concept 1")
        #
        axs[1].scatter(X_plt[n_data1:, 0], X_plt[n_data1:, 1], c=y_true[n_data1:])
        axs[1].grid(True)
        axs[1].set_ylabel("concept 2")
        plt.show()

    return X, y_true


def insert_label_noise(y_tru, noise_str="1sided", noise_level=0.2, rnd_seed=0):
    """Insert label noise.
    It is NOT really used in our synthetic experiments as the label noise is indirectly generated
    due to verification latency. 2022/5/5

    Parameters:
        noise_str - clean, 2sided or 1sided
        noise_level - [0, 1]

    2022-3-10 Set this func
    """
    rng = check_random_state(rnd_seed)

    n_data, n_c0, n_c1 = len(y_tru), np.sum(y_tru == 0), np.sum(y_tru == 1)
    nb_noise = round(noise_level * n_data)

    # insert label noise
    y_obv = np.copy(y_tru)
    if noise_str.lower() == "2sided":
        noise_ids = rng.choice(n_data, size=nb_noise, replace=False)
        y_obv[noise_ids] = 1 - y_tru[noise_ids]
    elif noise_str.lower() == "1sided":
        if n_c1 >= nb_noise:
            noise_id_c1 = rng.choice(n_c1, size=nb_noise, replace=False)
            y_obv[np.where(y_tru == 1)[0][noise_id_c1]] = 0
        else:
            raise Exception("%s: #(noisy X_org) should be no less then #(minority X_org)" % noise_str)
    else:
        raise Exception("Non-defined noise_str=%s" % noise_str)
    return y_obv


def delayed_labeling(y_tru, mean_delay_steps=20, rnd_seed=0):
    """
    Simulate delayed labeling as JIT-SDP via Gamma distribution to simulate verification latency in time steps.
    Gamma(shape, scale) has two parameters, where "mean = shape*scale" and "var = shape*scale^2"
    In this study, we set "shape = 2" so that Gamma(shape=2, scale=*) is not too skewed,

    See the orange line with k=2 in https://en.wikipedia.org/wiki/Gamma_distribution
    See NB-foundation-p87-N077.

    2022/4/1 created by Liyan Song
    """
    my_rng = check_random_state(seed=rnd_seed)

    shape = 2  # Gamma distribution
    scale = mean_delay_steps / shape
    ind_c1, ind_c0 = np.where(y_tru == 1)[0], np.where(y_tru == 0)[0]
    VLs = np.zeros(y_tru.shape)

    delay_case = "1sided".lower()
    if delay_case == "1sided".lower():
        VLs[ind_c1] = my_rng.gamma(shape, scale, ind_c1.shape[0])
    elif delay_case == "2sided".lower():
        VLs = my_rng.gamma(shape, scale, y_tru.shape[0])
    return VLs


class my_norm_scaler:
    """set up my normaliser for DenStream
    Note that the 1st fea "fix_bug" does not get involved & we should have 13 fea-s.
    2021-12-4   create
    2022-1-18   move back into main_test_cl_DenStream()
    """
    def __init__(self, n_fea=2, norm_name="z_score"):
        self.n_fea = n_fea
        self.norm_name = norm_name  # by default z-score
        #
        if self.norm_name.lower() == "min_max".lower():
            self.norm_scaler = preprocessing.MinMaxScaler()
        elif self.norm_name.lower() == "z_score".lower():
            self.norm_scaler = preprocessing.StandardScaler()

    def check_feature(self, XX):
        assert XX.shape[1] == self.n_fea, "#fea should be 13. PLS check if it only has one X_org."

    def my_fit(self, XX):
        # optimally, fit normaliser with all X_org XX
        self.check_feature(XX)
        return self.norm_scaler.fit(XX)  # note: "fix_bug" NOT normalized

    def my_transform(self, xx):
        if xx.ndim == 1:  # if xx contains only 1 X_org sample
            xx = xx.reshape((-1, self.n_fea))
        return self.norm_scaler.transform(xx)
        # 9 May 2022 specified in jit-sdp, the 1st dim remains unchanged.
        # return np.hstack((xx[:, 0].reshape(-1, 1), self.norm_scaler.transform(xx[:, 1:])))


def set_train_stream_syn(prev_test_time, curr_test_time, new_data, data_buffer=None, wait_steps=10):
    """ set training stream of the cross-project X_org
    :para new_data: numpy, (n_sample, 16-Xs) for which the 16-Xs are (time, 13-fea, vl, y)

    2022-4-2    Liyan creates this method by copying from real_data_stream()
    """
    # data_stream info
    idx_time, idx_vl, idx_y = 0, -2, -1
    if new_data.ndim == 1:  # debug
        new_data = new_data.reshape((1, -1))

    """store new_data into data_buffer (time, 13-fea, vl, y)
    """
    for dd in range(new_data.shape[0]):
        data_1new = new_data[dd, :].reshape((1, -1))
        # VIP overwrite clean X_org's VL to np.inf
        if data_1new[0, idx_y] == 0:
            data_1new[0, idx_vl] = np.inf
        # set data_buffer, (ts, XX, vl)
        if data_buffer is None:  # initialize
            data_buffer = data_1new
        else:
            data_buffer = np.vstack((data_buffer, data_1new))

    """create and update the training sets
    Consider VL and label noise: if there are labeled training XX becomes available 
    after last time and until current time, set the defect and the clean X_org sets, 
    and maintain the data_buffer carefully."""
    # 1) set train_data_defect and update data_buffer
    is_defect = curr_test_time > (data_buffer[:, idx_time] + data_buffer[:, idx_vl])
    train_data_defect = data_buffer[is_defect, :]
    # update data_buffer: pop out defect-inducing X_org
    data_buffer = data_buffer[~is_defect, :]  # (time, 13-fea, vl, y)

    # 2) set train_data_clean and update data_buffer
    wait_days_clean_upp = curr_test_time > data_buffer[:, idx_time] + wait_steps
    wait_days_clean_low = prev_test_time <= data_buffer[:, idx_time] + wait_steps
    wait_days_clean = wait_days_clean_low & wait_days_clean_upp
    train_data_clean = data_buffer[wait_days_clean, :]  # possible label noise

    # VIP update data_buffer: pop out the 'real' clean X_org
    actual_clean = data_buffer[:, idx_y] == 0
    # actual_clean = np.isinf(data_buffer[:, ID_vl])  # NOTE clean X_org's VL should have been assigned to np.inf
    wait_actual_clean = wait_days_clean & actual_clean
    data_buffer = data_buffer[~wait_actual_clean, :]  # (ts, 2-fea, vl, y)

    # 3) set train_data_unlabeled, no need to update data_buffer
    idx_upp_time_unlabeled = data_buffer[:, idx_time] < curr_test_time
    lowest_time = max(prev_test_time, curr_test_time - wait_steps)
    idx_low_time_unlabeled = data_buffer[:, idx_time] >= lowest_time
    train_data_unlabeled = data_buffer[idx_upp_time_unlabeled & idx_low_time_unlabeled]

    return data_buffer, train_data_defect, train_data_clean, train_data_unlabeled


if __name__ == "__main__":
    # # synthetic X_org stream with class imbalance
    gen_syn_data_stream()
    # syn_name, drift_name = "Agrawal", "abrupt"
    # n_data, tst_c1_percent, rnd_seed = 3000, 0.2, 0
    # X, y_true = gen_syn_data_stream(syn_name, drift_name, n_data, tst_c1_percent, rnd_seed, is_plot=True)

    # # label noise
    # noise_str, noise_level, noise_seed = "1sided", 0.2, 1
    # y_obv = insert_label_noise(y_true, noise_str, noise_level, noise_seed)
    #
    # # delayed labeling
    # VLs = delayed_labeling(y_true, 10, 0)

