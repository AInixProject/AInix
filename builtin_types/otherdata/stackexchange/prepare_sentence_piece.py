import argparse
import multiprocessing
from typing import List
import sentencepiece
import tempfile
import os


def train_sentence_piece(args) -> str:
    with tempfile.TemporaryDirectory() as tempd:
        prefix = os.path.join(tempd, "spm_m")
        sentencepiece.SentencePieceTrainer.Train(
            f'--input={args.srcsentences} --model_prefix={prefix} '
            f'--vocab_size={args.vocabsize} '
            f'--num_threads={multiprocessing.cpu_count()} '
            f'--max_sentence_length=6000k '
            f'--input_sentence_size=1000000'
        )
        with open(prefix + ".vocab") as outf:
            data = outf.read()
    return data




def process_sentence_pieces(data: str) -> List[str]:
    # Split by line and strip out the log probability
    toks = [
        r.split("\t")[0]
        for r in data.split("\n")
    ]
    # remove off the unk, sos, and eos
    toks = toks[3:]
    toks = _strip_whitespace_char(toks)
    return toks


def _strip_whitespace_char(toks: List[str]) -> List[str]:
    # remove the _ thing
    seen = set()
    new_toks = []
    for t in toks:
        if not t:
            continue
        if t[0] == '▁':
            t = t[1:]
        if not t:
            continue
        if t in seen:
            continue
        new_toks.append(t)
        seen.add(t)
    return new_toks


def _write_to_vocab_file(toks: List[str], dest_path: str):
    print(f"saving to {dest_path}")
    with open(dest_path, "w") as f:
        f.write("\n".join(toks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--srcsentences')
    parser.add_argument('-o', '--outfile')
    parser.add_argument('-n', '--vocabsize', type=int, default=1000)
    args = parser.parse_args()
    data = train_sentence_piece(args)
    toks = process_sentence_pieces(data)
    _write_to_vocab_file(toks, args.outfile)
    print("done.")
