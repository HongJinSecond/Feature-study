import os
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp


# global variables
flnm_input_format = "./%s_data.csv"
dir_output_format = "./%s.output/"


def do_friedman_test(ctrl_id=0, key_pf="recall-1"):
    """
    Friedman test with the pairwise post-hoc Conover comparisons given rejection of H0.
        Other post-hoc tests can refer to https://scikit-posthocs.readthedocs.io/en/latest/posthocs_api/
    The adopted effect size is Kendall's W, based on Cohen's interpretation as
        - 0.1 - small effect; 0.3 - moderate; >0.5 - large effect.
    Note: the PF is accuracy-like, such as gmean and accuracy, the larger, the better.
        For error-like PF, we need to adapt the codes later on. todo

    Refer to the below websites:
        https://www.reneshbedre.com/blog/friedman-test-python.html
        https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_conover_friedman/
    TC on 2022/12/17, Liyan on 2022/12/18
    """
    test_name = "friedman"
    flnm_input = flnm_input_format % test_name
    dir_output = dir_output_format % test_name
    os.makedirs(dir_output, exist_ok=True)
    to_name_pre = "to"

    """prepare data_long.pd"""
    # load data.csv
    data_pd = pd.read_csv(flnm_input)
    # obtain data keys
    key_data = data_pd.columns[0]  # data_pd.keys()[0]
    key_clf = "method"
    clf_name_pd = data_pd.keys()[1:]

    # prepare data_long_pd to run friedman.python
    data_long_pd = pd.melt(data_pd.reset_index(), id_vars=[key_data], value_vars=clf_name_pd)
    data_long_pd.columns = [key_data, key_clf, key_pf]  # set pd_long column names

    """step 1. run friedman test"""
    alpha_sig = 0.05
    out_friedman = pg.friedman(data=data_long_pd, dv=key_pf, within=key_clf, subject=key_data)
    p_value = out_friedman["p-unc"].values[0]
    reject_H0 = (p_value < alpha_sig)
    effect_kendall = out_friedman["W"].values[0]  # Kendall's W
    effect_interpret = "small"  # default
    if 0.3 < effect_kendall <= 0.5:
        effect_interpret = "moderate"
    elif 0.5 < effect_kendall:
        effect_interpret = "large"
    # output 1. friedman test
    to_see1_pd = pd.DataFrame({'reject-H0': [reject_H0], 'p-value': [p_value],
                              'effect_kendall': [effect_kendall], 'effect_interpret': effect_interpret},
                              index=["value"])
    to_see1_pd.to_csv(dir_output + to_name_pre + "1-friedman.csv")
    print("friedman test:\n ", to_see1_pd)

    """step 2. run pairwise post-hoc Conover comparisons"""
    post_p_value_pd = sp.posthoc_conover_friedman(
        a=data_long_pd, y_col=key_pf, group_col=key_clf, block_col=key_data, p_adjust="fdr_bh", melted=True)
    post_rejectH0_pd = (post_p_value_pd < alpha_sig)
    # output 2. all post-hoc-s
    post_p_value_pd.to_csv(dir_output + to_name_pre + "2_1-post_p_val.csv")
    post_rejectH0_pd.to_csv(dir_output + "to2_2-post_rejH0.csv")

    """step 3. for the control method"""
    # clf rankings & ave-rank across datasets
    clf_ranks_pd = data_pd.iloc[:, 1:].rank(axis=1, ascending=False)
    clf_ave_rank_pd = clf_ranks_pd.mean(axis=0)
    # output 3. for the control method
    assert ctrl_id <= post_p_value_pd.keys().size, "control ID should be within # methods"
    ctrl_name = post_p_value_pd.keys()[ctrl_id]  # auto
    ctrl_post_p_val_pd = post_p_value_pd.loc[ctrl_name]
    ctrl_post_rejH0_pd = post_rejectH0_pd.loc[ctrl_name]
    col_names = ["ave-rank", "p-value", "post-reject"]
    rslt_2see_pd = pd.concat((clf_ave_rank_pd, ctrl_post_p_val_pd, ctrl_post_rejH0_pd), axis=1, keys=col_names).T
    rslt_2see_pd.to_csv(dir_output + to_name_pre + "3-control_method.csv")
    print("post-hoc tests:\n", rslt_2see_pd)
    print("stat rslt.csv have been saved to %s." % dir_output)


if __name__ == "__main__":
    do_friedman_test()
