"""Install package."""

import io
import re

from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def read_version(filepath: str) -> str:
    """Read the __version__ variable from the file.

    Args:
        filepath: probably the path to the root __init__.py

    Returns:
        the version
    """
    match = re.search(
        r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        io.open(filepath, encoding="utf_8_sig").read(),
    )
    if match is None:
        raise SystemExit("Version number not found.")
    return match.group(1)


# ease installation during development
vcs = re.compile(r"(git|svn|hg|bzr)\+")
try:
    with open("requirements.txt") as fp:
        VCS_REQUIREMENTS = [
            str(requirement)
            for requirement in parse_requirements(fp)
            if vcs.search(str(requirement))
        ]
except FileNotFoundError:
    # requires verbose flags to show
    print("requirements.txt not found.")
    VCS_REQUIREMENTS = []

# TODO: Update these values according to the name of the module.
setup(
    name="signature_sampling",
    version=read_version("signature_sampling/__init__.py"),  # single place for version
    description="Installable signature_sampling package. Modify as needed.",
    long_description=open("README.md").read(),
    url="https://github.ibm.com/NJA/signature_sampling",
    author="Nikita Janakarajan",
    author_email="nja@zurich.ibm.com",
    # the following exclusion is to prevent shipping of tests.
    # if you do include them, add pytest to the required packages.
    packages=find_packages(".", exclude=["*tests*"]),
    package_data={"signature_sampling": ["py.typed"]},
    extras_require={
        "vcs": VCS_REQUIREMENTS,
        "test": ["pytest", "pytest-cov"],
        "dev": [
            # tests
            "pytest",
            "pytest-cov",
            # checks
            "black==21.5b0",
            "flake8",
            "mypy",
        ],
    },
    install_requires=[
        # versions should be very loose here, just exclude unsuitable versions
        # because your dependencies also have dependencies and so on ...
        # being too strict here will make dependency resolution harder
        "click",
    ],
)
