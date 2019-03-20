#!/usr/bin/env bash
# To run and save the result, use: sh test_script.sh | tee log_modified-curvature-motion_niter=20.txt

for i in `seq 10 2 30`;
do
    python3 test.py --d modified_curvature_motion --p 0.9 $i
    # sleep 1h
done

