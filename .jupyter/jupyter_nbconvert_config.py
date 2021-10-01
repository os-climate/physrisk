# type: ignore
# flake8: noqa

"""Configuration file for Jupyter NB Convert."""

## Set the log level by value or name.
c.Application.log_level = "DEBUG"

## The date format used by logging formatters for %(asctime)s
c.Application.log_datefmt = "%Y-%m-%dT%H:%M:%S"

## The Logging format template
# Let's at least try for an logstash format
c.Application.log_format = (
    "{"
    '"@timestamp": "%(asctime)s", '
    '"@version": 1, '
    '"level": "%(levelname)s", '
    '"name": "%(name)s", '
    '"message": "%(message)s"'
    "}"
)

## Writer class used to write the  results of the conversion
c.NbConvertApp.writer_class = "FilesWriter"

## The time to wait (in seconds) for output from executions. If a cell execution
#  takes longer, an exception (TimeoutError on python 3+, RuntimeError on python
#  2) is raised.
#
#  `None` or `-1` will disable the timeout. If `timeout_func` is set, it
#  overrides `timeout`.
c.ExecutePreprocessor.timeout = 1200

## The export format to be used, either one of the built-in formats ['asciidoc',
#  'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst',
#  'script', 'slides'] or a dotted object name that represents the import path
#  for an `Exporter` class
c.NbConvertApp.export_format = "notebook"

## Executes all the cells in a notebook
c.ExecutePreprocessor.enabled = True

## Name of kernel to use to execute the cells. If not set, use the kernel_spec
#  embedded in the notebook.
c.ExecutePreprocessor.kernel_name = "python3"

## Automation specific settings
import os

if os.getenv("RUN_IN_AUTOMATION"):
    from pathlib import Path

    ## Directory to write output(s) to. Defaults to output to the directory of each
    #  notebook. To recover previous default behaviour (outputting to the current
    #  working directory) use . as the flag value.
    base_dir = Path(os.getenv("LOCAL_DATA_PATH"))
    notebook_dir = Path(os.getenv("NOTEBOOK_NAME")).parent
    c.FilesWriter.build_directory = str(base_dir / "notebooks" / notebook_dir)

    ## Whether to apply a suffix prior to the extension (only relevant when
    #  converting to notebook format). The suffix is determined by the exporter, and
    #  is usually '.nbconvert'.
    c.NbConvertApp.use_output_suffix = False
