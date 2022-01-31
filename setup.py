from setuptools import setup
import os, codecs

def get_version():
    dir = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(dir, "VERSION"), 'r') as fp:
        return fp.read()


def get_requirements():
    with open("requirements.in") as reqs:
        return [
            line.split()[0]
            for line in reqs.read().splitlines()
            if not line.startswith("#")
        ]


setup(
    version = get_version(),
    install_requires=get_requirements(),
)
