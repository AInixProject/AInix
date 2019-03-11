import torch
from typing import List

from ainix_common.parsing.model_specific.tokenizers import ModifiedStringToken, \
    ModifiedWordPieceTokenizer, StringTokensMetadata
from ainix_kernel.model_util.stringops import get_all_words

# Adapted from https://github.com/TellinaTool/nl2bash/blob/master/nlp_tools/constants.py
# and from http://xpo6.com/list-of-english-stop-words/
STOP_WORDS = frozenset({
    "a",
    "an",
    "the",
    "be",
    "is",
    "been",
    "being",
    "was",
    "were",
    "are",
    "has",
    "have",
    "had",
    # "here",
    "there",
    "do",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "me",
    "my",
    "myself",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "must",
    "should",
    "would",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    #"here",
    #"here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "she",
    "she'd",
    "she'll",
    "she's",
    "it",
    "it's",
    "its",
    "itself",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "let",
    "let's",
    "this",
    "that",
    "these",
    "those",
    "what",
    "what's",
    "which",
    "whose",
    "how",
    "how's"
    
    # Others
    "of",
    "get",
    "to"
})


def get_non_stop_word_mask(
    tokens: List[ModifiedWordPieceTokenizer],
    metad: StringTokensMetadata,
    stop_words = STOP_WORDS,
    pad_to_len = None
) -> torch.Tensor:
    """For a sequence tokens, gets a tensor of dim (seq_len, ) which
    is 1 if NOT a stop word and 0 if it is a stop word or a pad"""
    mask = torch.ones((pad_to_len or len(tokens), ))
    mask[len(tokens):] = 0
    for word, (start_i, end_i) in get_all_words(tokens, metad):
        if word.lower() in stop_words:
            mask[start_i:end_i] = 0
    return mask


def get_non_stop_word_mask_batched(
    token_batch: List[List[ModifiedWordPieceTokenizer]],
    metad_batch: List[StringTokensMetadata]
) -> torch.Tensor:
    if len(token_batch) != len(metad_batch):
        raise ValueError("Should be equal")
    pad_to_len = max(map(len, token_batch))
    return torch.stack(tuple([
        get_non_stop_word_mask(toks, meta, pad_to_len=pad_to_len)
        for toks, meta in zip(token_batch, metad_batch)
    ]))
