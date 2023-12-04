#!/bin/sh

# set -x

if [ $# -ne 1 ]; then
    echo "Usage:	$0 [test folder]"; exit 1
elif [ ! -d "$1" ]; then
    echo "Error: specified target was not a folder"; exit 1
else
    # Target specified was a folder
    TARGET="$1"
fi

# GIT move/rename files test_{NAME}.py to {NAME}_test.py
for TEST in $(find "$TARGET"/* -name "test*.py" -print0 | xargs); do
    PATHNAME=$(dirname "$TEST")
    FILENAME=$(basename "$TEST")
    PREFIX_STRIP="${FILENAME#*_}"
    NEW_TEST=$(echo "$PREFIX_STRIP" | sed "s/.py/_test.py/")
    echo "Renaming $TEST -> $PATHNAME/$NEW_TEST"
    git mv "${TEST}" "$PATHNAME"/"${NEW_TEST}"
done
