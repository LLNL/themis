import time

from themis import user_utils


def prep_run():
    if user_utils.run_id() <= 10:
        while True:
            time.sleep(0.5)
    return
