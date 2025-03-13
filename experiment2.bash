#!/bin/bash

res=(5 10 20 40)
data=(50 100 250 500 1000 2000)
epochs=(10 20 50 100 200 300 400)

for r in "${res[@]}"; do
    for d in "${data[@]}"; do
        for e in "${epochs[@]}"; do
            echo "res=${r}, ndata=${d}, epochs=${e}"
            ckpt="checkpoints/2d_res${r}/${d}/ep${e}.pth"
            python scripts/designer.py --dim 2 --res ${r} --ckpt ${ckpt} --task task2
        done
    done

done
