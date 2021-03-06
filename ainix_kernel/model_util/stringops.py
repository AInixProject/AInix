"""Operations on string-like things which is useful in models"""
from typing import List, Tuple

import torch

from ainix_common.parsing.model_specific import parse_constants
from ainix_common.parsing.model_specific.tokenizers import ModifiedStringToken, WhitespaceModifier, \
    CasingModifier, StringTokensMetadata
import numpy as np


def get_word_lens_of_moded_tokens(
    token_batch: List[List[ModifiedStringToken]],
    pad_to_len: int = None
) -> torch.Tensor:
    """Given a batch of tokens, creates a tensor that is the length of the contigous
    non-whitespace characters a token is part of.

    For example [["_my", "_name", "'s", "_John", "_Jing", "le", "hiemer", "_."]] would
    encode as   [[ 1   ,  2     ,  2  ,  1     ,  3     ,  3  ,  3      ,  1  ]]

    By default it will pad with a value 0
    to be the same as the longest token length, unless you specify a pading length.
    Alternatively if the tokens already includes parsing_contants.PAD, the len there
    will be set as 0.
    """
    pad_to_len = max(map(len, token_batch)) if pad_to_len is None else pad_to_len
    all_lens = []
    for tokens in token_batch:
        word_lens = np.zeros((pad_to_len,))
        cur_len = 0
        word_start_pointer = None
        for i in range(len(tokens)):
            cur = tokens[i]
            is_pad = cur.token_string == parse_constants.PAD
            new_word_start = cur.whitespace_modifier == WhitespaceModifier.AFTER_SPACE_OR_SOS
            if new_word_start or is_pad:
                if word_start_pointer is not None:
                    word_lens[word_start_pointer:i] = cur_len
                    cur_len = 0
                word_start_pointer = i
                if is_pad:
                    break
            cur_len += 1
        else:
            # Reached the end without seeing a pad.
            # We need to set the length for the last word
            word_lens[word_start_pointer:len(tokens)] = cur_len
        all_lens.append(word_lens)
    return torch.tensor(all_lens).float()


def get_all_words(
    tokens: List[ModifiedStringToken],
    metad: StringTokensMetadata
) -> List[Tuple[str, Tuple[int, int]]]:
    """Get's all the whitespace sperated words in a tokenization.
    Returns:
        A list of tuples with values:
            word: the string word
            start, end: a tuple of the start and end index of a slice into
                the tokens that make up that word. End index is python style
                exclusive.
    """
    word_start_i = None
    words = []
    for i, tok in enumerate(tokens):
        if tok.whitespace_modifier == WhitespaceModifier.AFTER_SPACE_OR_SOS:
            join_start_i = metad.actual_pos_to_joinable_pos[word_start_i] if word_start_i else None
            join_end_i = metad.actual_pos_to_joinable_pos[i-1]
            if join_start_i is not None and join_end_i is not None:
                joinable_toks = metad.joinable_tokens[join_start_i:join_end_i+1]
                joined_str = "".join(joinable_toks).strip()
                words.append((joined_str, (word_start_i, i)))
            if tok.token_string == parse_constants.PAD:
                break
            word_start_i = i
    return words
