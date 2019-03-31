#!/usr/bin/env bash


mkdir combined
fname="./combined/sentences.txt"
rm ${fname}
for network in $(cat networks.txt); do
    cat ${network}-stackexchange/sentences.txt >> ${fname}
    echo >> ${fname}
done

cat redditstuff/reddit_stuff.txt >> ${fname}

# Make other versions of the sentences
cd combined
sed '/^$/d' sentences.txt > sentences_no_blank_lines.txt
tr '[:upper:]' '[:lower:]' < sentences_no_blank_lines.txt > sentences_lower.txt
cd ..


