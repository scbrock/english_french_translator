#!/bin/bash

source ./myenv/bin/activate

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs_local/cuda-10.0/lib64

export HOME=/w/246/sbrock

#python3 en_fr.py
python3 translate.py < input.txt
#python3 emb.py
#pip3 list

deactivate

