#!/usr/bin/env python

import argparse
import socket

from mpi4py import MPI


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multiply a value by the number of total MPI "
            "ranks and write the result to a file."
        )
    )
    parser.add_argument("val", type=int, help="The value to multiply")
    parser.add_argument(
        "--outfile",
        "-o",
        help="The outfile to write the result",
        default="mpi_app_output.txt",
    )
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    print(
        "Hello world from {}, rank {} of {}".format(
            socket.gethostname(), rank, comm_size
        )
    )
    if rank == 0:
        with open(args.outfile, "w") as file_handle:
            file_handle.write(
                "input:\n{}\noutput:\n{}\n".format(args.val, args.val * comm_size)
            )


if __name__ == "__main__":
    main()
