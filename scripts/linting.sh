#!/bin/bash

npm install eslint @babel/core @babel/eslint-parser --save-dev
echo "Run with: eslint --ext .toml ."
pre-commit install
pre-commit autoupdate
