import codecs
import os

from setuptools import setup


def get_version():
    dir = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(dir, "VERSION"), "r") as fp:
        return fp.read()


def get_requirements():
    with open("requirements.in") as f:
        return f.read().splitlines()


setup(
    version=get_version(),
    install_requires=get_requirements(),
)
