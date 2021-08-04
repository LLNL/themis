import csv

from themis import user_utils


def post_run():
    result = []
    with open("my_script_out.csv") as file_handle:
        reader = csv.reader(file_handle)
        for row in reader:
            result.extend([float(val) for val in row])
    return result
