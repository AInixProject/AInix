#!/usr/bin/env bash

# Downloads and extracts different stack exchanges
for network in $(cat networks.txt); do
    # Don't redownload things we have already done.
    if [ -f ${network}-stackexchange/Posts.xml ]; then
        echo "Already found files ${network}. Delete them to redownload"
        continue
    fi
    mkdir ${network}-stackexchange
    cd ${network}-stackexchange
    wget -O ${network}.7z https://archive.org/download/stackexchange/${network}.com.7z
    7z x ${network}.7z
    ls | grep -v Posts.xml | xargs rm
    cd ..
done
