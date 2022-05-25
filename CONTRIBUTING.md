# Contributing to physrisk

## Getting started
To get set up, clone and enter the repo.
```
git clone git@github.com:os-climate/physrisk.git
cd physrisk
```

We recommend using [pipenv](https://pipenv.pypa.io/en/latest/) for a
consistent working environment.
```
pip install pipenv
pipenv install
pipenv shell
```

When adding a package for use in new or improved functionality,
`pipenv install <package-name>`. Or, when adding something helpful for
testing or development, `pipenv install -d <package-name>`.

## Development
Patches may be contributed via pull requests to
https://github.com/os-climate/physrisk.

All changes must pass the automated test suite, along with various static
checks.

[Black](https://black.readthedocs.io/) code style and
[isort](https://pycqa.github.io/isort/) import ordering are enforced
and enabling automatic formatting via [pre-commit](https://pre-commit.com/)
is recommended:
```
pre-commit install
```

To ensure compliance with static check tools, developers may wish to run black and isort against modified files.

E.g.,
```
# auto-sort imports
isort .
# auto-format code
black .
```

Code can then be tested using tox.
```
# run static checks and unit tests
tox
# run only tests
tox -e py3
# run only static checks
tox -e static
# run unit tests and produce an HTML code coverage report (/htmlcov)
tox -e cov
```

## IDE set-up
For those using VS Code, configure tests ('Python: Configure Tests') to use 'pytest'
to allow running of tests within the IDE.

## Releasing
Actions are configured to release to PyPI on pushing a tag. In order to do this:
- Update VERSION
- Create new annotated tag and push 
```
git tag -a v1.0.0 -m "v1.0.0"
git push --follow-tags
```

## Forking workflow
This is a useful clarification of the forking workflow:
https://gist.github.com/Chaser324/ce0505fbed06b947d962

## Project Organization
------------

    ├── LICENSE
    ├── Pipfile            <- Pipfile stating package configuration as used by Pipenv.
    ├── Pipfile.lock       <- Pipfile.lock stating a pinned down software stack with as used by Pipenv.
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── methodology        <- Contains LaTeX methodology document
    │    └── literature    <- Literature review
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. These comprise notebooks used for on-boarding
    │                         hazard data, on-boarding vulnerability models and tutorial
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │   ├── physrisk       <- physrisk source code
    │   ├── test           <- physrisk tests; follows same folder structure as physrisk
    │   └── visualization  <- Deprecated visualizations, migrated to physrisk-ui)
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
