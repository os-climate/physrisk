# Contributing to physrisk

## Getting started

To get set up, clone and enter the repo.

```bash
git clone https://github.com/os-climate/physrisk.git
cd physrisk
```

We recommend using [uv](https://docs.astral.sh/uv/) for maintaining a consistent working environment.
There are a number of [installation options](https://docs.astral.sh/uv/getting-started/installation/).
Note that an advantage of uv is that it can also be used to maintain python installations
(via ```uv python install```) and select the Python installation be be used for the creation of the
project's virtual environment, e.g. ```uv python pin 3.11```.

The command

```bash
uv sync
```

will create a virtual environment (.venv folder in the project
folder) and install the dependencies.
We recommend that the IDE workspace use this virtual environment when
developing.

When adding a package for use in new or improved functionality,
`uv add <package-name>`. Or, when adding something for
development, `uv add --dev <package-name>`.

## Development

Patches may be contributed via pull requests to
<https://github.com/os-climate/physrisk>.

All changes must pass the automated test suite, along with various static
checks. Tests and static checks can be run via the commands:

```bash
uv run pytest
```

```bash
uv run pre-commit run --all-files
```

Other checks are then run with Actions within GitHub.

## IDE set-up

For those using VS Code, configure tests ('Python: Configure Tests') to
use 'pytest' to allow running of tests within the IDE.

## Building documentation

Building of the documentation relies on Pandoc which must be [installed]((https://pandoc.org/installing.html))
in order to build the documentation locally. One this is done, to build:

```bash
cd docs
make html
```

and to open in a browser:

```bash
open ./_build/html/index.html
```

## Releasing

Actions are configured to release to PyPI on pushing a tag. In order to
do this:

- Ensure version in pyproject.toml is updated (will require pull request
  like any other change)
- Create new annotated tag and push

```bash
git tag -a v1.0.0 -m "v1.0.0"
git push --follow-tags
```

## Forking workflow

This is a useful clarification of the forking workflow:
<https://gist.github.com/Chaser324/ce0505fbed06b947d962>

## Project Organization

---

```text
├── LICENSE
    ├── pdm.lock           <- pdm.lock stating a pinned down software stack
    │                         as used by pdm.
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── methodology        <- Contains LaTeX methodology document.
    │    └── literature    <- Literature review.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org
    │                         for details.
    │
    ├── notebooks          <- Jupyter notebooks. These comprise notebooks used
    │                         for on-boarding hazard data, on-boarding
    │                         vulnerability models and tutorial.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .)
    │                         so src can be imported.
    │
    ├── src                <- Source code for use in this project.
    │   └── physrisk       <- physrisk source code.
    │    
    ├── tests              <- physrisk tests; follows same folder structure as physrisk.
    │
    ├── pyproject.toml     <- central location of project settings.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io.

---
