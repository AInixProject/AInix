"""
Codes for processing stuff from cookiemonster

Parts of this code is adapted from huggingface/pytorch-pretrained-BERT
https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_lm_finetuning.py
That was derived from stuff from Google AI and NVIDIA CORPORATION.
Their code is avaiable under Apache 2.0 license on an "AS IS" BASIS
"""
import torch
from typing import Tuple, List, Iterable, MutableMapping

from torch.utils.data import Dataset, DataLoader

from ainix_common.parsing.model_specific import tokenizers, parse_constants
from ainix_common.parsing.model_specific.parse_constants import TASK_TOK
from ainix_common.parsing.model_specific.tokenizers import StringTokenizerWithMods, \
    ModifiedStringToken, StringTokensMetadata, CasingModifier, WhitespaceModifier, apply_case_mod
from tqdm import tqdm
import random
import torch.nn.functional as F

import attr

from ainix_kernel.model_util.vocab import torchify_moded_tokens, Vocab, BasicVocab


class CookieMonsterDataset(Dataset):
    def __init__(
        self,
        corpuses: List[str],
        tokenizer: StringTokenizerWithMods,
        vocab: Vocab,
        max_docs_to_load: int = 9e9,
        use_cuda: bool = False
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
        self.max_docs_to_load = max_docs_to_load
        self.data = list(map(self._load_corpus, corpuses))
        self.num_senteces = sum(sum(len(d) - 1 for d in c) for c in self.data)
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        #print(f"avg sentence len {sum(self.sentence_lens) / len(self.sentence_lens)}")

    def _load_sentence(self, sentence_text: str) -> Tuple[int, str]:
        #toks, metad = self.tokenizer.tokenize(sentence_text)
        #tok_len = len(toks)
        tok_len = None  # Don't actually use it, so quicker
        self.sentence_lens.append(tok_len)
        return tok_len, sentence_text

    def _load_doc(self, doc_text: str) -> List[Tuple[int, str]]:
        sentences = doc_text.split("\n")
        return list(map(self._load_sentence, sentences))

    def _load_corpus_text(self, corpus_text: str) -> List[List[Tuple[int, str]]]:
        doc_split = corpus_text.split("\n\n")
        to_take = min(self.max_docs_to_load, len(doc_split))
        return [self._load_doc(dt)
                for dt in tqdm(doc_split[:to_take], unit="documents")]

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
        tokens: List[ModifiedStringToken],
        token_metad: StringTokensMetadata,
        change_prob: float = 0.15,
        if_change_prob_mask: float = 0.8
    ) -> Tuple[
        List[ModifiedStringToken],
        StringTokensMetadata,
        List[Tuple[int, ModifiedStringToken]]
    ]:
        restore_map: List[Tuple[int, ModifiedStringToken]] = []
        for i, (tok, joinable_pos) in enumerate(
                zip(tokens, token_metad.actual_pos_to_joinable_pos)):
            prob = random.random()
            # Randomly mask with some probability (default 15%)
            if tok.token_string not in parse_constants.ALL_SPECIALS and prob < change_prob:
                prob /= change_prob
                if prob < if_change_prob_mask:
                    # 80% randomly change token to mask token (we use task token, not mask though)
                    tokens[i] = ModifiedStringToken(
                        TASK_TOK, CasingModifier.CASELESS, tokens[i].whitespace_modifier)
                    token_metad.joinable_tokens[joinable_pos] = TASK_TOK
                elif prob < 0.9:
                    # 10% randomly change token to random token
                    rand_tok = self.vocab.index_to_token(random.randint(0, len(self.vocab) - 1))
                    if rand_tok not in parse_constants.ALL_SPECIALS:
                        has_case = rand_tok.upper() != rand_tok
                        rand_case = random.choice(
                            (CasingModifier.LOWER, CasingModifier.FIRST_UPPER,
                             CasingModifier.ALL_UPPER)) if has_case else CasingModifier.CASELESS
                        rand_whitespace = tokens[i].whitespace_modifier if random.random() < 0.75 \
                            else random.choice(list(WhitespaceModifier))
                        tokens[i] = ModifiedStringToken(
                            rand_tok, rand_case, rand_whitespace)
                        token_metad.joinable_tokens[joinable_pos] = apply_case_mod(
                            rand_tok, rand_case)
                        # clean up a change in whitespace
                        if rand_whitespace == WhitespaceModifier.NOT_AFTER_SPACE and \
                                joinable_pos > 0 and \
                                token_metad.joinable_tokens[joinable_pos - 1] == " ":
                            token_metad.joinable_tokens[joinable_pos - 1] = ""
                # -> rest 10% randomly keep current token
                restore_map.append((i, tok))
        return tokens, token_metad, restore_map

    def random_sample_toks(self) -> Tuple[
        List[ModifiedStringToken],
        StringTokensMetadata,
        bool,
        Tuple[int, ModifiedStringToken]
    ]:
        (first_sent, second_sent), was_sequential = self._get_two_sentences()
        combined_sent = first_sent + f" {parse_constants.TASK_SPLITTER} " + second_sent
        tokens, metad = self.tokenizer.tokenize(combined_sent)
        tokens, metad, restore_map = self._random_mask(tokens, metad)
        return tokens, metad, was_sequential, restore_map

    def random_sample_sentence_str(self) -> Tuple[str, bool]:
        tokens, metad, was_sequential, restore_map = self.random_sample_toks()
        # We don't return a restore mask, as it might not line anymore after we
        # a token was replaced with a random word which is now able to be joined
        # with another token
        return "".join(metad.joinable_tokens), was_sequential

    def torchify_example(
        self,
        tokens: List[ModifiedStringToken],
        was_seq: bool,
        restore_map: Tuple[int, ModifiedStringToken]
    ) -> 'LMExampleTorched':
        token_inds, case_inds, whitespace_inds = torchify_moded_tokens(
            tokens, self.vocab, device=self.device)
        out = LMExampleTorched(
            tokens=token_inds,
            token_case_mod=case_inds,
            token_whitespace_mod=whitespace_inds,
            is_sequential=was_seq,
            mask_inds=torch.tensor([ind for ind, real_token in restore_map],
                                   device=self.device),
            mask_expected_ind=self.vocab.token_seq_to_indices(
                [real_token.token_string for ind, real_token in restore_map]),
            mask_expected_case=[real_token.casing_modifier.value for ind, real_token in restore_map]
        )
        return out

    def random_sample(self) -> 'LMExampleTorched':
        tokens, metad, was_seq, restore_map = self.random_sample_toks()
        return self.torchify_example(tokens, was_seq, restore_map)

    def __getitem__(self, index) -> 'LMExampleTorched':
        # TODO (Make this deterministic based off ind
        return self.random_sample()

    def __len__(self):
        return self.num_senteces


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

    def __len__(self):
        return len(self.tokens)


@attr.s(auto_attribs=True, frozen=True)
class LMBatch:
    tokens: torch.LongTensor
    token_case_mod: torch.LongTensor
    token_whitespace_mod: torch.LongTensor
    original_token_counts: torch.LongTensor  # originoal length before padding each of batch
    is_sequential: torch.Tensor
    mask_inds: List[torch.LongTensor]  # The indices into tokens for masks tokens
    mask_expected_ind: List[torch.LongTensor]
    mask_expected_case: List[torch.LongTensor]

    @classmethod
    def from_example_list(
        cls,
        examples: List[LMExampleTorched],
        pad_ind: int,
        device=torch.device("cpu")
    ) -> 'LMBatch':
        lens = list(map(len, examples))
        targ_len_tokens = max(lens)

        def pad_toks(t: torch.Tensor, val=pad_ind) -> torch.Tensor:
            return F.pad(t, pad=(0, targ_len_tokens - len(t)), value=val)

        out = LMBatch(
            tokens=torch.stack([pad_toks(e.tokens) for e in examples]),
            token_case_mod=torch.stack(
                [pad_toks(e.token_case_mod, CasingModifier.CASELESS) for e in examples]),
            token_whitespace_mod=torch.stack(
                [pad_toks(e.token_whitespace_mod, WhitespaceModifier.AFTER_SPACE_OR_SOS)
                 for e in examples]),
            original_token_counts=torch.tensor(lens, device=device),
            is_sequential=torch.tensor([e.is_sequential for e in examples],
                                       device=device),
            mask_inds=[e.mask_inds for e in examples],
            mask_expected_ind=[e.mask_expected_ind for e in examples],
            mask_expected_case=[e.mask_expected_case for e in examples]
        )
        return out


class CookieMonsterBatchIterator:
    def __init__(
        self,
        dataset: CookieMonsterDataset,
        batch_size,
        bucket_count=2,
        max_num_batches=None
    ):
        # TODO (DNGros): figure out how to use data loader so can use fancy
        # stuff like multiprocessing
        # self.loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_count = bucket_count
        self.example_q = []
        self.max_num_batches = max_num_batches
        self.yielded_batches = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.max_num_batches and self.yielded_batches > self.max_num_batches:
            raise StopIteration()
        while len(self.example_q) < self.batch_size * self.bucket_count:
            self.example_q.append(self.dataset.random_sample())
        self.example_q.sort(key=lambda x: len(x), reverse=True)
        start = random.randint(0, self.bucket_count - 1) * self.batch_size
        end = start + self.batch_size
        take_examples = self.example_q[start:end]
        self.example_q = self.example_q[:start] + self.example_q[end:]
        self.yielded_batches += 1
        return LMBatch.from_example_list(
            take_examples, self.dataset.vocab.token_to_index(parse_constants.PAD),
            device=self.dataset.device
        )


def human_test(dataset: CookieMonsterDataset, samples):
    rights = []
    for i in range(samples):
        sent, gt_seq = dataset.random_sample_sentence_str()
        guess = input(f"TRY(1 seq/ 0 not): {sent}")
        right = bool(float(guess)) == gt_seq
        print("guess", bool(float(guess)), "actual", gt_seq, "right", right)
        rights.append(right)
    print(f"Right percent {sum(rights) / samples * 100}")


if __name__ == "__main__":
    tokenizer, vocab = tokenizers.get_default_pieced_tokenizer_word_list()
    dataset = CookieMonsterDataset(
        ["../../../builtin_types/otherdata/stackexchange/unix-stackexchange/sentences.txt"],
        tokenizer, BasicVocab(vocab + parse_constants.ALL_SPECIALS), max_docs_to_load=100
    )
    print(len(dataset))
    print(dataset.random_sample())
    for batch in CookieMonsterBatchIterator(dataset, batch_size=4, max_num_batches=2):
        print(batch)
    human_test(dataset, 10)
