Tools for extracting pretraining data based off stack exchange questions.

Note after downloading this expands to ~2GB. But we only keep
about 500MB.

We get data from https://archive.org/details/stackexchange
which is availble under [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

First you should use download_data.sh
Then you should split those into sentences using split_stacke_data.py

Example / recreation notes.
This is just a cluttery spewing of commands that were used while making this 
the first time.
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
# This MIGHT work. I actually had to install it from source because it wasn't getting
# added onto my path correctly.
# There are pretty easy instructions for this at https://github.com/google/sentencepiece

# Actually use it
# Make version with no blank lines and lowercase
$ sed '/^$/d' sentences.txt > sentences_no_blank_lines.txt
$ tr '[:upper:]' '[:lower:]' < sentences_no_blank_lines.txt > sentences_lower.txt
$ python3 prepare_sentence_piece.py --srcsentences unix-stackexchange/sentences_lower.txt --outfile unix_vocab.txt -n 3000
# Or if you want to just run it raw
$ spm_train --input sentences_no_blank_lines.txt --model_prefix testmod --vocab_size 1000 --num_threads 8 --max_sentence_length 6000k

# random notes about experiements with fast text
# follow install instructions at https://fasttext.cc/docs/en/supervised-tutorial.html
# Tokenize the stuff
$ cat sentences_no_blank_lines.txt | spm_encode --model=testmod_upper_2000.model > sentences_tokenized_with_upper_2000.txt
# train (disable subword stuff though with maxn = 0)
$ $PATH_TO_FASTTEXT skipgram -input sentences_tokenized_with_upper_2000.txt -output fasttext/m3 -thread 8 -maxn 0


```