#!/bin/bash

#set -x

FILEPATH="pyproject.toml"

for TAG in $(git tag -l | sort | uniq); do
echo "" > /dev/null
done
echo "Version string from tags: ${TAG}"

echo "Performing string substitution on: ${FILEPATH}"
sed -i "s/.*version =.*/version = \"$TAG\"/" "${FILEPATH}"
echo "Versioning set to:"
grep version "${FILEPATH}"
echo "Script completed!"; exit 0
