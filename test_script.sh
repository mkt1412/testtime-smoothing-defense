#!/usr/bin/env bash
# To run and save the result, use: sh test_script.sh | tee log.txt

for i in `seq 0.5 0.1 1.5`;
do
    python3 test.py $i
done

