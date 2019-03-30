"""A script for going from a stack exchange dump file into a
BERT-like file where you have an output which is a bunch of
line-delimented sentences with a empty line between files"""
import argparse
import xml.etree.ElementTree
from typing import Optional, List
import re
import html
import attr
from tqdm import tqdm
DEFAULT_NAME = "./unix-stackexchange/small_posts.xml"
MIN_NUM_OF_WORDS_IN_SENTENCE = 4 # 4
MIN_NUM_CHARS_IN_SENTENCE = 16 # 16
MAX_SENTENCE_CHARACTERS = 320
MAX_CODE_PRE_LEN = 160  # Exclude posts with really long code blocks


def get_post_body(row_element, min_score=-9e9) -> Optional[str]:
    for key, val in row_element.items():
        if key == "Score":
            if int(val) < min_score:
                return None
        if key == "Body":
            return val
    raise ValueError()


html_tag_re = re.compile(r'<.*?>')
code_pre_tag = re.compile(r'<pre><code>.*?</code></pre>', re.DOTALL | re.MULTILINE)


def remove_html_tags(string: str) -> str:
    return re.sub(html_tag_re, '', string)


def longest_code_pre_len(string: str) -> Optional[int]:
    matches = code_pre_tag.findall(string)
    if not matches:
        return None
    return max(map(len, matches)) - len("<pre><code></code></pre>")


def clean_post_body(body: str) -> Optional[str]:
    longest_quoted_out_code_block = longest_code_pre_len(body)
    if longest_quoted_out_code_block is not None and \
            longest_quoted_out_code_block > MAX_CODE_PRE_LEN:
        return None
    body = remove_html_tags(body)
    body = unscape_html_entities(body)
    return body

# Make a regex to match sentences.
#   This regex is pretty nasty.
#   Basically first we match a non-whitespace (\S) and as little as we can (.+?)
#   Then we use negative lookahead to look for
#       sentence ending punctuation followed by whitespace or end of line
#       or EOF
#       or a new line
# This fancyness allows us handle sentences with filenames or decimal numbers
sentence_pattern = r'\S.+?(?:[.?!]+(?=\s|$)|(?=\Z)|(?=\n))'
sentence_regex = re.compile(sentence_pattern, re.DOTALL | re.MULTILINE)

def sentence_split(body: str):
    return [str(s).lstrip() for s in sentence_regex.findall(body)]


def unscape_html_entities(string: str):
    return html.unescape(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--srcfile', default=DEFAULT_NAME)
    parser.add_argument('-o', '--outfile')
    args = parser.parse_args()
    print(f"Loading {args.srcfile}")
    e = xml.etree.ElementTree.parse(args.srcfile).getroot()
    docs: List[List[str]] = []
    print("Splitting sentences.")
    for row in tqdm(e):
        doc_strs = []
        body = get_post_body(row, -1)
        if body is None:
            continue
        body = clean_post_body(body)
        if body is None:
            continue
        sentences = sentence_split(body)
        if len(sentences) < 2:
            continue
        shortest_char_count_sentence = min(map(len, sentences))
        if shortest_char_count_sentence < MIN_NUM_CHARS_IN_SENTENCE:
            continue
        longest_char_count_sentence = max(map(len, sentences))
        if longest_char_count_sentence > MAX_SENTENCE_CHARACTERS:
            continue
        shortest_word_count_sentence = min(len(s.split()) for s in sentences)
        if shortest_word_count_sentence < MIN_NUM_OF_WORDS_IN_SENTENCE:
            continue
        docs.append(sentences)
    out_str = "\n\n".join(("\n".join(d) for d in docs))
    if args.outfile:
        print(f"Writing to {args.outfile}")
        with open(args.outfile, "w") as text_file:
            text_file.write(out_str)
    else:
        print(out_str)
