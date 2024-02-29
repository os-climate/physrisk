#!/bin/bash

#set -x

FILEPATH="pyproject.toml"

grep "version.*=" "$FILEPATH" | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3