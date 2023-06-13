#!/bin/bash

for i in {0..29} # 29 is inclusive in bash
do
   sbatch submit_code.sh ig.py $i
   echo ig.py $i submitted
done
