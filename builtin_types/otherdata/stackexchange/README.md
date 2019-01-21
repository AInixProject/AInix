Tools for extracting pretraining data based off stack exchange questions.

Note after downloading this expands to ~2GB. But we only keep
about 500MB.

We get data from https://archive.org/details/stackexchange
which is availble under [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

First you should use download_data.sh
Then you should split those into sentences using split_stacke_data.py

Example / recretion notes.
This is just a cluttery spewing of commands
that were used while making this the first time.
Depending on what you are doing you might not need all these.
```bash
# Download data from stack exchange archive
$ ./download_data.sh
# Split it into a txt file which has a sentence per line 
# and an empty line between documents (this is like what bert wants)
$ python3 split_stacke_data.py -s unix-stackexchange/Posts.xml -o unix-stackexchange/sentences.txt

# Train a word tokenizer

# Download google/sentencepiece
$ pip3 install sentencepiece
# Actually use it


```