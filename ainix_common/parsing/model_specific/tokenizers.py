"""Classes for tokenizeing a input string. Note this is not used by the actual
string parsers. Rather it can be useful for something like models which wish
to tokenize the input string. It is used in string parsers in order to enable
producing AST's with copying."""
import os
from abc import ABC, abstractmethod
from typing import Iterable, Generator, List, Tuple, Hashable, Union, Optional, Dict, \
    MutableMapping, TypeVar

import attr
import pygtrie

from ainix_common.parsing.model_specific.parse_constants import TOKEN_SPECIALS
from ainix_common.parsing.typecontext import AInixObject, AInixType
from ainix_common.parsing.model_specific import parse_constants
from ainix_common.parsing.ast_components import AstNode, ObjectNode, ObjectChoiceNode
from ainix_common.parsing import ast_components
import numpy as np
from itertools import chain
import functools
from enum import IntEnum, unique


class Tokenizer(ABC):
    def __init__(self):
        self._vectorized_tokenize = np.vectorize(self.tokenize)
        self._vectorized_tokenize_first = np.vectorize(lambda x: self.tokenize(x)[0])

    @abstractmethod
    def tokenize(self, to_tokenize) -> Tuple[List[Hashable], List]:
        """

        Args:
            to_tokenize: The object to tokenize. Could be a str, or depending
                on the tokenizer, some other type.

        Returns:
            A tuple. The first element is a list of of the individual tokens.
            The second is metadata about those tokens which could vary depending
            on the parser. It is intended for something like pointers of back to
            the location of the tokens or something like that.
        """
        pass

    def tokenize_batch(self, batch, take_only_tokens: bool = False) -> List:
        """Tokenize multiple values at once

        Args:
            batch: Sequence of objects to tokenize
            take_only_tokens: self.tokenize is expected to return a tuple of
                (tokens, metadata). If this is set to true, then we only take
                the tokens, ignoring the metadata.

        Returns:
            List with self.tokenize applied on every element. The result
            will be tuples if the take_only_tokens is false, and sequence
            if it is true.
        """
        if take_only_tokens:
            return [self.tokenize(s)[0] for s in batch]
        else:
            return [self.tokenize(s) for s in batch]

    def get_save_state_dict(self) -> Dict:
        raise NotImplemented()


class StringTokenizer(Tokenizer):
    def tokenize(self, to_tokenize: str) -> Tuple[List[str], 'StringTokensMetadata']:
        """Returns Tuple. The first is a list of the actual tokens that could
        go into a vocab. The second is metadata about that. See StringTokensMetadata"""
        raise NotImplemented()


class StringTokenizerWithMods(Tokenizer):
    def tokenize(
        self,
        to_tokenize: str
    ) -> Tuple[List['ModifiedStringToken'], 'StringTokensMetadata']:
        raise NotImplemented()


@attr.s(frozen=True)
class StringTokensMetadata:
    """Metadata about the tokens returned by a StringTokenizer

    Args:
        joinable_tokens: A list of tokens which when joined with empty string
            gets back to the origional input string. The tokenization process
            might add in special tokens like "<SPACE>", "<SOS>", or some
            customness for inner-word splitting. This is used for
            something like copying which is only allowed to perform copies
            on token boundries.
        joinable_tokens_pos_to_actual: A mapping from the joinable tokens
            back to original tokens. For example if origional tokens were
            ["<SOS>", "foo", "<SPACE>", "_b", "ar"] joinable tokens might be
            ["foo", " ", "b", "ar"] and the map might be [1, 2, 3, 4]. If
            mapping is None it is considered there is no direct mapping for
            this token and it cannot serve as the start or end of a copy.
        actual_pos_to_joinable_pos: The inverse of the above arg
    """
    joinable_tokens = attr.ib(type=List[str])
    # TODO (DNGros): Change these maps to callable functions. Having them as
    # lists is kinda excessive when sometimes have a closedform map
    joinable_tokens_pos_to_actual = attr.ib(type=List[Optional[int]])
    actual_pos_to_joinable_pos = attr.ib(type=List[Optional[int]])

    @joinable_tokens_pos_to_actual.default
    def fac(self):
        return list(range(len(self.joinable_tokens)))

    @actual_pos_to_joinable_pos.default
    def ofac(self):
        return list(range(len(self.joinable_tokens)))

    def __attrs_post_init__(self):
        assert len(self.joinable_tokens) == len(self.joinable_tokens_pos_to_actual)


@unique
class CasingModifier(IntEnum):
    LOWER = 0
    ALL_UPPER = 1
    FIRST_UPPER = 2
    CASELESS = 3  # True if all symbols
    SINGLE_CHAR_UPPER = 4  # True if all symbols


@unique
class WhitespaceModifier(IntEnum):
    AFTER_SPACE_OR_SOS = 0
    NOT_AFTER_SPACE = 1


@attr.s(auto_attribs=True, frozen=True)
class ModifiedStringToken:
    token_string: str
    casing_modifier: CasingModifier
    whitespace_modifier: WhitespaceModifier


class NonLetterTokenizer(StringTokenizer):
    @functools.lru_cache(maxsize=10)  # Larger cache?
    def tokenize(self, to_tokenize: str) -> Tuple[List[str], StringTokensMetadata]:
        """Takes in a string and outputs a tokenization splitting at non-letter boundries"""
        if not isinstance(to_tokenize, str):
            raise ValueError(f"NonAsciiTokenizer expects string inputs. Got {to_tokenize}")
        out_tokens = [[]]
        for c in to_tokenize:
            char_is_letter = ('A' <= c <= 'z')
            if not char_is_letter:
                if out_tokens[-1]:
                    out_tokens.append([])

                if c == " ":
                    out_tokens[-1].append(parse_constants.SPACE)
                else:
                    out_tokens[-1].append(c)
                out_tokens.append([])
            else:
                out_tokens[-1].append(c)
        out_tokens = ["".join(toklist) for toklist in out_tokens if len(toklist) >= 1]
        replace_spaces = [x if x != parse_constants.SPACE else " " for x in out_tokens]
        metadata = StringTokensMetadata(replace_spaces)
        return out_tokens, metadata

    def get_save_state_dict(self) -> Dict:
        return {"name": "NonLetterTokenizer"}


# TODO (DNGros): This should be unified with the tokenizer in generic_strings.
class ModifiedWordPieceTokenizer(StringTokenizerWithMods):
    def __init__(self, vocab: List[str], merge_long_files: bool = True):
        super().__init__()
        self.trie: pygtrie.CharTrie[str, CasingModifier] = pygtrie.CharTrie()
        self.vocab_list = list(vocab)
        self.merge_long_files = merge_long_files
        for tok in vocab:
            assert tok.lower() == tok, "Vocab should be all lower case"
            tok_upper = tok.upper()
            is_casable = tok_upper != tok
            if is_casable:
                self.trie[tok_upper] = CasingModifier.ALL_UPPER if len(tok) > 1 \
                    else CasingModifier.SINGLE_CHAR_UPPER
            tok_first_cap = tok[0].upper() + tok[1:]
            if tok_first_cap != tok_upper:
                self.trie[tok_first_cap] = CasingModifier.FIRST_UPPER
            self.trie[tok] = CasingModifier.LOWER if is_casable else CasingModifier.CASELESS
        for special in TOKEN_SPECIALS:
            self.trie[special] = CasingModifier.CASELESS


    SOS_TOK = ModifiedStringToken(
        parse_constants.SOS,
        CasingModifier.CASELESS, WhitespaceModifier.AFTER_SPACE_OR_SOS)

    EOS_TOK = ModifiedStringToken(
        parse_constants.EOS,
        CasingModifier.CASELESS, WhitespaceModifier.AFTER_SPACE_OR_SOS)

    @functools.lru_cache(maxsize=50)
    def tokenize(
        self,
        to_tokenize: str
    ) -> Tuple[List[ModifiedStringToken], StringTokensMetadata]:
        outs_strs: List[ModifiedStringToken] = []
        joinable_tokens: List[str] = []
        joinable_tokens_to_actual: List[Optional[int]] = []
        actual_to_joinable_ind: List[Optional[int]] = []
        cur_ind = 0
        after_whitespace = True

        # Handle SOS
        outs_strs.append(self.SOS_TOK)
        actual_to_joinable_ind.append(None)

        # Go through and parse tokens
        while cur_ind < len(to_tokenize):
            if to_tokenize[cur_ind] == " ":
                # TODO (DNGros): Extra white space might have semantic meaning
                # we should allow this to be captured and reconstructed.
                # It might be wise to ignore the extra whitespace if not in quotes
                # and captures it when in quotes.
                after_whitespace = True
                joinable_tokens.append(" ")
                cur_ind += 1
                joinable_tokens_to_actual.append(None)
                continue

            cur_str = to_tokenize[cur_ind:]
            longest_prefix = self.trie.longest_prefix(cur_str)
            is_an_unk_tok = not longest_prefix
            if is_an_unk_tok:
                raw_tok: str = cur_str[0]
                token_str = parse_constants.UNK
                casing_mod = CasingModifier.CASELESS
            else:
                raw_tok: str = longest_prefix.key
                casing_mod: CasingModifier = longest_prefix.value
                token_str = raw_tok.lower() if casing_mod != CasingModifier.CASELESS else raw_tok

            outs_strs.append(ModifiedStringToken(
                token_string=token_str,
                casing_modifier=casing_mod,
                whitespace_modifier=WhitespaceModifier.AFTER_SPACE_OR_SOS if after_whitespace \
                    else WhitespaceModifier.NOT_AFTER_SPACE
            ))
            actual_to_joinable_ind.append(len(joinable_tokens))
            joinable_tokens_to_actual.append(len(outs_strs) - 1)
            joinable_tokens.append(raw_tok)
            cur_ind += len(raw_tok)
            after_whitespace = False

        # Handle EOS
        outs_strs.append(self.EOS_TOK)
        actual_to_joinable_ind.append(None)

        assert len(outs_strs) == len(actual_to_joinable_ind)
        metadata = StringTokensMetadata(
            joinable_tokens, joinable_tokens_to_actual, actual_to_joinable_ind)
        if self.merge_long_files:
            merge_tokens(outs_strs, metadata)
        return outs_strs, metadata

    def get_save_state_dict(self):
        return {"version": 0, "name": type(self).__name__, "vocab": self.vocab_list}


def apply_case_mod(string: str, case_mod: CasingModifier):
    if case_mod == CasingModifier.CASELESS:
        assert string.lower() == string.upper()
    elif case_mod == CasingModifier.ALL_UPPER:
        return string.upper()
    elif case_mod == CasingModifier.FIRST_UPPER:
        return string[0].upper() + string[1:]
    elif case_mod == CasingModifier.LOWER:
        return string.lower()
    else:
        raise ValueError()


MOD_TOK_FOR_MERGE = ModifiedStringToken(
    parse_constants.MERGED_TOK, CasingModifier.CASELESS, WhitespaceModifier.NOT_AFTER_SPACE)


def merge_tokens(
    modtokens: List[ModifiedStringToken],
    metad: StringTokensMetadata
) -> None:
    """Looks for things that look like file names and merges their center tokens
    so that way the files don't contain so many tokens and create noise.

    This mutates the input deleting the middle tokens of filenames replacing
    them with just one <MERGE_TOK> token.
    """
    word_start_pointer = None
    i = 0
    while i < len(modtokens) + 1:
        # We iterate through the len + 1 in order to still hit the new_word_start
        # condition for the last word in the string
        cur = modtokens[i] if i < len(modtokens) else None
        new_word_start = cur is None or \
                         cur.whitespace_modifier == WhitespaceModifier.AFTER_SPACE_OR_SOS
        if new_word_start:
            if word_start_pointer is not None:
                join_start = metad.actual_pos_to_joinable_pos[word_start_pointer]
                join_end = metad.actual_pos_to_joinable_pos[i-1]
                if join_start is None or join_end is None:
                    word_start_pointer = i
                    i += 1
                    continue
                joinable_toks = metad.joinable_tokens[join_start:join_end+1]
                joined_str = "".join(joinable_toks).strip()
                long_enough_to_merge = i - word_start_pointer > 3
                if long_enough_to_merge and looks_like_a_file(joined_str):
                    # We set the 2nd token in the word to be the merge
                    modtokens[word_start_pointer + 1] = MOD_TOK_FOR_MERGE
                    metad.joinable_tokens[join_start + 1] = "".join(
                        metad.joinable_tokens[join_start + 1: join_end])
                    for _ in range(word_start_pointer + 2, i - 1):
                        # Delete everything after the 2nd token (the <MERGE_TOK>) but before
                        # the last token in the word. We always delete at word_start + 2 as that
                        # keeps shifting as we delete.
                        del modtokens[word_start_pointer + 2]
                        del metad.actual_pos_to_joinable_pos[word_start_pointer + 2]
                        del metad.joinable_tokens[join_start + 2]
                        del metad.joinable_tokens_pos_to_actual[join_start + 2]
                    # Iterate through and decrease the mapping inds of everything in front of what
                    # we deleted.
                    num_we_deleted = i - word_start_pointer - 3
                    for dec_ind in range(word_start_pointer + 2, len(modtokens)):
                        if metad.actual_pos_to_joinable_pos[dec_ind] is not None:
                            metad.actual_pos_to_joinable_pos[dec_ind] -= num_we_deleted
                    for dec_ind in range(join_start + 2,
                                         len(metad.joinable_tokens_pos_to_actual)):
                        if metad.joinable_tokens_pos_to_actual[dec_ind] is not None:
                            metad.joinable_tokens_pos_to_actual[dec_ind] -= num_we_deleted
                    i -= num_we_deleted
            word_start_pointer = i
        i += 1


class SpaceTokenizer(StringTokenizer):
    def tokenize(self, to_tokenize: str) -> Tuple[List[str], StringTokensMetadata]:
        tokens = to_tokenize.split()
        joinable = [tokens[0]] + list(chain.from_iterable(((" ", t) for t in tokens[1:])))
        joinable_to_actual = [0] + list(chain.from_iterable(
            ((None, i + 1) for i in range(len(tokens[1:])))
        ))
        actual_to_joinable = list(range(0, len(tokens)*2, 2))
        return to_tokenize.split(), StringTokensMetadata(
            joinable, joinable_to_actual, actual_to_joinable)


class AstStringTokenizer(Tokenizer):
    def tokenize(self, to_tokenize: AstNode) -> Tuple[List[str], List[AstNode]]:
        out_tokens = []
        out_nodes = []
        for pointer in to_tokenize.depth_first_iter():
            node = pointer.cur_node
            if isinstance(node, ObjectNode):
                out_tokens.append(ast_components.indexable_repr_object(node.implementation.name))
            elif isinstance(node, ObjectChoiceNode):
                out_tokens.append(ast_components.indexable_repr_classify_type(
                    node.get_type_to_choose_name()))
            else:
                raise ValueError(f"Unrecognized node {node}")
            out_nodes.append(node)
        return out_tokens, out_nodes


class AstValTokenizer(Tokenizer):
    def tokenize(
        self,
        to_tokenize: AstNode
    ) -> Tuple[List[Union[AInixObject, AInixType]], List[AstNode]]:
        out_tokens = []
        out_nodes = []
        for pointer in to_tokenize.depth_first_iter():
            node = pointer.cur_node
            if isinstance(node, ObjectNode):
                out_tokens.append(node.implementation)
            elif isinstance(node, ObjectChoiceNode):
                out_tokens.append(node.type_to_choose)
            else:
                raise ValueError(f"Unrecognized node {node}")
            out_nodes.append(node)
        return out_tokens, out_nodes


T = TypeVar('T')


def add_pad_arbitrary(token_seqs: List[List[T]], pad_with: T) -> Tuple[List[List[T]], List[int]]:
    """Add padding tokens to an collection of tokenized values so all the same len

    Returns:
        padded_seqs: new vals but padded.
        origional_lengths
    """
    lengths = list(map(len, token_seqs))
    longest_len = max(lengths)
    pad_val_arr = [pad_with]
    padded_seqs = [existing + pad_val_arr*(longest_len - length)
                   for existing, length in zip(token_seqs, lengths)]
    return padded_seqs, lengths


def add_str_pads(token_seqs: List[List[str]], pad_with=parse_constants.PAD):
    return add_pad_arbitrary(token_seqs, pad_with)


def add_pads_to_mod_tokens(
    token_seqs: List[List[ModifiedStringToken]]
):
    return add_pad_arbitrary(
        token_seqs,
        ModifiedStringToken(
            parse_constants.PAD,
            CasingModifier.CASELESS,
            WhitespaceModifier.AFTER_SPACE_OR_SOS
        )
    )


def tokenizer_from_save_dict(save_dict: dict):
    name = save_dict['name']
    if name == "NonLetterTokenizer":
        return NonLetterTokenizer()
    if name == ModifiedWordPieceTokenizer.__name__:
        return ModifiedWordPieceTokenizer(save_dict['vocab'])
    else:
        raise ValueError(f"Bad name {name}")


def get_default_pieced_tokenizer_word_list() -> Tuple[StringTokenizerWithMods, List[str]]:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "unix_vocab.txt")) as f:
        vocab_str = f.read()
    vocab = vocab_str.split("\n")
    return ModifiedWordPieceTokenizer(vocab), vocab


common_exts = (".sh", ".py", ".txt", ".tar.gz", ".png", ".tmp", ".xml", ".zip",
               ".csv", ".cpp", ".h", 'html', '.go', '.sql', '.json', '.java')


def looks_like_a_file(string: str):
    """A super hacky heuristic function to guess if a string looks like a path"""
    if string.startswith("s/"):
        # Could be a sed expression. This is admittedly a crappy detection of this...
        return False
    non_filey_chars = ("?", "@", "!", "(", ")", "//")
    for c in non_filey_chars:
        if c in string:
            return False
    for ext in common_exts:
        if string.endswith(ext) and (len(string) - len(ext)) >= 3:
            return True
    if string.startswith("./") or string.startswith("../"):
        return True
    if string.count("/") >= 2:
        return True
    return False


