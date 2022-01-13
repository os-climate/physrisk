from setuptools import setup
import os, codecs

def get_version():
    dir = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(dir, "VERSION"), 'r') as fp:
        return fp.read()

setup(version = get_version())