from themis import user_utils


def prep_run():
    """Add an argument to the run command"""
    return "./" + user_utils.run().sample["materials"] + ".sh"
