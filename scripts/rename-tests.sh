#!/bin/bash

#set -x

for FILE in $(find . -name "*test_*" -print | xargs -0 echo); do
    if (echo "$FILE" | grep -v ".zip" > /dev/null)
    then
        NEW_NAME=$(echo "$FILE" | sed "s:test_::g" | sed "s:.py:_test.py:g")
        echo "Renaming: $FILE to $NEW_NAME"
        git mv "$FILE" "$NEW_NAME"
    else
        echo "Skipping: $FILE"
    fi
done
