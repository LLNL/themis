"""Run all of the integration tests."""

import functools

import utils


if __name__ == "__main__":
    load_tests = functools.partial(utils.load_tests_func, ("integration",))
    utils.main("integration")
