#!/bin/bash
script_dir="$( cd "$( dirname "${BACH_SOURCE[0]}" )" && pwd )"
python ${script_dir}/quickstary.py >> /out/output.txt
if [[ $? > 0 ]]; then
    echo "Error"
    exit 1
fi
echo "Success"