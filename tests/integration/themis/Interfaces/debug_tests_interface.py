import pickle

from themis import user_utils


def prep_ensemble():
    thm = user_utils.themis_handle()
    completed_runs = thm.count_by_status(thm.RUN_SUCCESS)
    with open("debug_tests_prep_ensemble.pkl", "wb") as file_handle:
        pickle.dump(completed_runs, file_handle)


def prep_run():
    return str(10 * user_utils.run_id())


def post_run():
    return 10 * user_utils.run().sample["foo"]


def post_ensemble():
    thm = user_utils.themis_handle()
    completed_runs = thm.count_by_status(thm.RUN_SUCCESS)
    with open("debug_tests_post_ensemble.pkl", "wb") as file_handle:
        pickle.dump(completed_runs, file_handle)
