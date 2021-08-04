"""Simulate an application that only succeeds after a couple of restarts."""


import os
import sys

RESTART_FILE = "restarts.txt"


def write_restarts_file(index):
    with open(RESTART_FILE, "w") as file_handle:
        file_handle.write(str(index))


def read_restarts_file():
    with open(RESTART_FILE, "r") as file_handle:
        return int(file_handle.readlines()[0])


def main():
    if not os.path.exists(RESTART_FILE):
        restart_index = 0
    else:
        restart_index = read_restarts_file()
    if restart_index < 1:
        write_restarts_file(restart_index + 1)
        print("Failure!")
        # exit with an error
        sys.exit(1)
    else:
        print("Success!")
        sys.exit(0)


if __name__ == "__main__":
    main()
