import torch
from typing import Tuple, List, Iterable, MutableMapping

from torch.utils.data import Dataset

from ainix_common.parsing.model_specific import tokenizers, parse_constants
from ainix_common.parsing.model_specific.tokenizers import StringTokenizerWithMods, \
    ModifiedStringToken, StringTokensMetadata
from tqdm import tqdm
import random

import attr

from ainix_kernel.model_util.vocab import torchify_moded_tokens, Vocab, BasicVocab


class CookieMonsterDataset(Dataset):
    def __init__(
        self,
        corpuses: List[str],
        tokenizer: StringTokenizerWithMods,
        vocab: Vocab
    ):
        assert len(corpuses) == 1, "Only one corpus currently supported"
        self.vocab = vocab
        # The data is in the a list which is indexed as
        # data[corpus][document_num][sentence_in_doc_num]
        # This gives a tuple.
        #   The first int is number of tokens the string will parse to
        #   The second element in the tuple is the actual sentence
        self.data: List[List[List[Tuple[int, str]]]] = []
        self.tokenizer = tokenizer
        self.sentence_lens = []
        self.data = list(map(self._load_corpus, corpuses))
        self.num_senteces = sum(sum(len(d) - 1 for d in c) for c in self.data)
        print(f"avg sentence len {sum(self.sentence_lens) / len(self.sentence_lens)}")

    def _load_sentence(self, sentence_text: str) -> Tuple[int, str]:
        toks, metad = self.tokenizer.tokenize(sentence_text)
        tok_len = len(toks)
        self.sentence_lens.append(tok_len)
        return tok_len, sentence_text

    def _load_doc(self, doc_text: str) -> List[Tuple[int, str]]:
        sentences = doc_text.split("\n")
        return list(map(self._load_sentence, sentences))

    def _load_corpus_text(self, corpus_text: str) -> List[List[Tuple[int, str]]]:
        doc_split = corpus_text.split("\n\n")
        return [self._load_doc(dt) for dt in tqdm(doc_split[:100], unit="documents")]

    def _load_corpus(self, path_to_corpus: str) -> List[List[Tuple[int, str]]]:
        with open(path_to_corpus, "r") as corp_file:
            corp_str = corp_file.read()
        return self._load_corpus_text(corp_str)

    def _get_two_sentences(self) -> Tuple[Tuple[str, str], int]:
        chosen_corpus = self.data[0]
        num_docs = len(chosen_corpus)
        chosen_doc_ind = random.randint(0, num_docs - 1)
        chosen_doc = chosen_corpus[chosen_doc_ind]
        want_sequential = random.random() < 0.5
        if want_sequential:
            first_sentence_ind = random.randint(0, len(chosen_doc) - 2)
            first_sentence = chosen_doc[first_sentence_ind]
            second_sentence = chosen_doc[first_sentence_ind + 1]
        else:
            first_sentence = random.choice(chosen_doc)
            other_doc_ind = (chosen_doc_ind + (len(chosen_corpus) - 1)) % len(chosen_corpus)
            other_doc = chosen_corpus[other_doc_ind]
            second_sentence = random.choice(other_doc)
        return (first_sentence[1], second_sentence[1]), want_sequential

    def _random_mask(
        self,
        sentence,
        mask_prob: float = 0.15
    ) -> Tuple[str, List[Tuple[int, ModifiedStringToken]]]:
        tokens, metad = self.tokenizer.tokenize(sentence)
        new_joinable = list(metad.joinable_tokens)
        restore_map: List[Tuple[int, ModifiedStringToken]] = []
        for i, (tok, joinable_pos) in enumerate(zip(tokens, metad.actual_pos_to_joinable_pos)):
            if tok.token_string not in parse_constants.ALL_SPECIALS and random.random() < mask_prob:
                new_joinable[joinable_pos] = parse_constants.TASK_TOK
                restore_map.append((i, tok))
        return "".join(new_joinable), restore_map

    def random_sample_sentence_str(self) -> Tuple[str, bool, Tuple[int, ModifiedStringToken]]:
        (first_sent, second_sent), was_sequential = self._get_two_sentences()
        combined_sent = first_sent + f" {parse_constants.TASK_SPLITTER} " + second_sent
        masked_sent, restore_map = self._random_mask(combined_sent)
        return masked_sent, was_sequential, restore_map

    def torchify_example(
        self,
        masked_sent: str,
        was_seq: bool,
        restore_map: Tuple[int, ModifiedStringToken]
    ) -> 'LMExampleTorched':
        # TODO (DNGros): Repeatedly doing the tokenization is terribly inefficient
        # should just tokenize once and keep things in tokenized form
        tokens, metad = self.tokenizer.tokenize(masked_sent)
        token_inds, case_inds, whitespace_inds = torchify_moded_tokens(tokens, self.vocab)
        return LMExampleTorched(
            tokens = token_inds,
            token_case_mod=case_inds,
            token_whitespace_mod=whitespace_inds,
            is_sequential=was_seq,
            mask_inds=torch.LongTensor([ind for ind, real_token in restore_map]),
            mask_expected_ind=self.vocab.token_seq_to_indices(
                [real_token.token_string for ind, real_token in restore_map]),
            mask_expected_case=[real_token.casing_modifier for ind, real_token in restore_map]
        )

    def random_sample(self) -> 'LMExampleTorched':
        masked_sent, was_seq, restore_map = self.random_sample_sentence_str()
        return self.torchify_example(masked_sent, was_seq, restore_map)

    def __getitem__(self, index) -> 'LMExampleTorched':
        # TODO (Make this deterministic based off ind
        return self.random_sample()

    def __len__(self):
        return self.num_senteces


def human_test(dataset: CookieMonsterDataset, samples):
    rights = []
    for i in range(samples):
        sent, gt_seq, restore_map = dataset.random_sample_sentence_str()
        guess = input(f"TRY(1 seq/ 0 not): {sent}")
        print(restore_map)
        right = bool(float(guess)) == gt_seq
        print("guess", bool(float(guess)), "actual", gt_seq, "right", right)
        rights.append(right)
    print(f"Right percent {sum(rights)/samples*100}")


@attr.s(auto_attribs=True, frozen=True)
class LMExampleTorched:
    tokens: torch.LongTensor
    token_case_mod: torch.LongTensor
    token_whitespace_mod: torch.LongTensor
    is_sequential: bool
    mask_inds: torch.LongTensor  # The indices into tokens for masks tokens
    mask_expected_ind: torch.LongTensor
    mask_expected_case: torch.LongTensor

    def __attrs_post_init__(self):
        assert len(self.tokens.shape) == 1
        assert len(self.tokens) == len(self.token_case_mod) == len(self.token_whitespace_mod)
        assert len(self.mask_inds.shape) == 1
        assert len(self.mask_inds) == len(self.mask_expected_case) == len(self.mask_expected_case)


@attr.s(auto_attribs=True, frozen=True)
class LMBatch:
    tokens: torch.LongTensor
    token_case_mod: torch.LongTensor
    token_whitespace_mod: torch.LongTensor
    is_sequential: torch.Tensor
    mask_inds: torch.LongTensor  # The indices into tokens for masks tokens
    mask_expected_ind: torch.LongTensor
    mask_expected_case: torch.LongTensor


if __name__ == "__main__":
    tokenizer, vocab = tokenizers.get_default_pieced_tokenizer_word_list()
    dataset = CookieMonsterDataset(
        ["../../../builtin_types/otherdata/stackexchange/unix-stackexchange/sentences.txt"],
        tokenizer, BasicVocab(vocab + parse_constants.ALL_SPECIALS)
    )
    print(len(dataset))
    print(dataset.random_sample())
    human_test(dataset, 10)
