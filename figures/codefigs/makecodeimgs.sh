#!/usr/bin/env bash

for file in $(ls *.yaml)
do
    echo ${file}
    pygmentize -f png -O style=lovelace,font_size=30 ${file} > ${file}.png
done
