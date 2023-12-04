#!/bin/bash

#set -x

for TAG in $(git tag -l | grep 202 | sort | uniq); do
git tag -d "${TAG}"git tag -d "$TAG"
done
echo "Script completed!"; exit 0
