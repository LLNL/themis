import os

import setuptools


_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def read_long_description():
    readme_path = os.path.join(
        _SCRIPT_DIR,
        "README.md"
    )
    with open(readme_path, "r") as fh:
        return fh.read()


if __name__ == "__main__":
    setuptools.setup(
        name="uqpipeline",
        version="0.0.1.dev1",
        author="LLNL UQP Team",
        author_email="uqpipeline-devs@llnl.gov",
        description="LLNL's UQ Pipeline",
        url="https://lc.llnl.gov/bitbucket/projects/UQP/repos/uqp/",
        long_description=read_long_description(),
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(
            where=_SCRIPT_DIR,
            include=["uqp" + ".*" * i for i in range(0, 5)] + ["themis" + ".*" * i for i in range(0, 5)]
        ),
        install_requires=[
            "numpy>=1.15",
            "scikit-learn",
            "scipy",
            "matplotlib",
        ],
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 2.7",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.6, >=2.7.16, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
        entry_points={
            "console_scripts": [
                "themis = themis.__main__:main",
                "themis-laf = themis.laf:main",
            ],
        },
    )

