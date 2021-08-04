from themis import user_utils
from themis import Run


def post_ensemble():
    """Add points until there are 50 total runs"""
    thm = user_utils.themis_handle()
    total_runs = thm.count_by_status()
    if total_runs < 50:
        num_runs_to_add = min(15, 50 - total_runs)
        thm.add_runs([Run({}, "0") for _ in range(num_runs_to_add)])
