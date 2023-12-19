#!/bin/bash

#set -x

FILEPATH="pyproject.toml"

if [ $# -ne 1 ] && [ $# -ne 0 ]; then
    echo "Usage: $0 [version-string]"
    echo "Substitutes the version string in pyproject.toml"; exit 1
elif [ $# -eq 1 ]; then
    VERSION=$1
    echo "Received version string: $VERSION"
else
    datetime=$(date +'%Y%m%d%H%M')
    pyver=$(python --version | awk '{print $2}')
    VERSION="${pyver}.${datetime}"
    echo "Defined version string: $VERSION"
fi

echo "Performing string substitution on: $FILEPATH"
sed -i "s/.*version =.*/version = \"$VERSION\"/" "$FILEPATH"
echo "Versioning set to:"
grep version "$FILEPATH"
echo "Script completed!"; exit 0
