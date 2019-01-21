#!/usr/bin/env bash
# Downloads and extracts unix stack exchange
mkdir unix-stackexchange
cd unix-stackexchange
wget -O unix.stackexchange.7z https://archive.org/download/stackexchange/unix.stackexchange.com.7z
7z x unix.stackexchange.7z
ls | grep -v Posts.xml | xargs rm
cd ..
