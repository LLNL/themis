import time
from decimal import Decimal
import os

from themis import user_utils
from themis import Run
from themis import utils


def post_run():
    with open("mpi_app_output.txt") as file_handle:
        lines = file_handle.readlines()
    return (int(lines[-1]), Decimal(lines[-1]))
