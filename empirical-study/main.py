from core_code import sdp_runs
from data_stream.real_data_stream import data_id_2name
from humla.humla import a_sdp_runs

def Run_Sdp_Model(clf_name="our", human_dict=None,
               project_id=0, nb_para_tune=1000, nb_test=-1, wait_days=15,
               seed_lst=range(50), verbose_int=1, is_plot=False, just_run=False, just_load=False):
    """
    this part we will run the models on the chosen project. U can change the path of datasets in data_stream/set_datastream.py

    the output can be checked in results/rslt.save
    :param clf_name: the model name
    :param human_dict: the hyperparameter for Humla
    :param project_id: the id for project, can be checked in data_stream/real_data_stream.py
    :param nb_para_tune: how many commits to auto para tune
    :param nb_test: how many commits for test, -1 means all
    :return:
    """

    assert clf_name in ["humla","oob","odasc","pbsa"],print(f"{clf_name} is not a valid command, please choose one from 'humla', 'oob' ,'odasc', 'pbsa'")
    print(f"-------------run project {data_id_2name(project_id)} in model {clf_name} -------------")
    if clf_name=="humla":
        a_sdp_runs("our",human_dict,project_id,nb_para_tune, nb_test, wait_days, seed_lst=seed_lst, verbose_int=verbose_int, is_plot=is_plot, just_run=just_run, just_load=just_load)
    else:
        sdp_runs(clf_name,project_id,nb_para_tune,nb_test,wait_days,seed_lst,verbose_int,pca_plot=is_plot,just_run=just_run,load_result=just_load)


if __name__ == '__main__':
    #clf_name : the name of the baseline model, you can choose the model below: ['oob','odasc','humla','pbsa']
    #project_id: the id of one project. That means you will choose the dataset of this id of project to run you experiment. you can find the id list in real_data_stream.py

    just_run = False
    just_load = False
    humla_opt = "humla"

    #parameter for HumLa
    human_dict= {
        "has_human": True, "human_err": 0, "human_eff": 1  # hyper-parameter for humla
    }


    #parameter for Eco-HumLa
    # human_dict = {
    #     "has_human": True, "human_err": 0, "human_eff": 1  # hyper-parameter for humla
    # }

    for i in range(1,23,1):
        Run_Sdp_Model(clf_name="humla",human_dict=human_dict,project_id=i)
