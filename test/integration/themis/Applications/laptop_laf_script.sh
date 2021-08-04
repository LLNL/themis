#!/bin/bash
set -e

INTEGER="%%integer%%"

if ! [[ "$INTEGER" =~ ^[0-9]+$ ]]
    then
        echo "Sorry integers only"
        exit 1
fi

exit 0
