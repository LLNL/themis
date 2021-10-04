"""Run all of the unit tests."""

import functools

import utils


if __name__ == "__main__":
    load_tests = functools.partial(utils.load_tests_func, ("unit",))
    utils.main("unit")
