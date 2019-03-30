#!/usr/bin/env bash

for network in $(cat networks.txt); do
    python3 split_stacke_data.py \
        -s ${network}-stackexchange/Posts.xml \
        -o ${network}-stackexchange/sentences.txt
done
