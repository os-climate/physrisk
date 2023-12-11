import os

from setuptools import setup


def get_version():
    dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(dir, "VERSION"), "r") as fp:
        return fp.read()


setup(version=get_version(), include_package_data=True)
