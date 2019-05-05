"""
Code for CookieMonster. This model is trained on a similar task to BERT. However,
it aims to be smaller, and, ideally, more efficient. Right now it is pretty
primitive though without the planned fancyness.

Parts of this code is adapted from huggingface/pytorch-pretrained-BERT
https://github.com/huggingface/pytorch-pretrained-BERT
    /blob/master/pytorch_pretrained_bert/modeling.py
That was derived from stuff from Google AI and NVIDIA CORPORATION.
Their code is avaiable under Apache 2.0 license on an "AS IS" BASIS
"""
import numpy as np
from typing import Optional, List, Tuple, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import math

from ainix_common.parsing.model_specific import tokenizers
from ainix_common.parsing.model_specific.tokenizers import CasingModifier, WhitespaceModifier, \
    ModifiedWordPieceTokenizer, ModifiedStringToken
from ainix_kernel.model_util.lm_task_processor.lm_set_process import LMBatch
from ainix_kernel.model_util.operations import pack_picks, avg_pool, GELUActivation, fastgelu
from ainix_kernel.model_util.stop_words import get_non_stop_word_mask_batched
from ainix_kernel.model_util.stringops import get_word_lens_of_moded_tokens
from ainix_kernel.model_util.usefulmodules import Conv1dSame
from ainix_kernel.model_util.vocab import Vocab, torchify_moded_tokens, BasicVocab, \
    torchify_batch_modded_tokens
from ainix_kernel.models.EncoderDecoder.encoders import RNNSeqEncoder, ModTokensEncoder, \
    QueryEncoder
from ainix_kernel.models.model_types import BertlikeLangModel
from ainix_kernel.multiembedder.multiencoder import Multiembedder
import ainix_kernel.model_util.transformer.layers


class CookieMonsterForPretraining(BertlikeLangModel):
    def __init__(
        self,
        base_encoder: 'ModTokensEncoder',
        use_cuda: bool = False
    ):
        self.base_encoder = base_encoder
        self.lm_head = CookieMonsterLMPredictionHead(base_encoder.get_tokens_input_weights())
        self.next_sent_head = NextSentenceCookieMonsterHead(
            self.base_encoder.output_size)
        #self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
        #    hidden_size_base, self.embedder.vocab_sizes[0], [10, 100, 1000])
        self.heads = nn.ModuleList([self.lm_head, self.next_sent_head])
        self.all_torch_models = nn.ModuleList([self.base_encoder, self.heads])
        self.optimizer: Optional[torch.optim.Optimizer] = None
        if use_cuda:
            self.all_torch_models.cuda()
        self.use_cuda = use_cuda

    def eval_run(
        self,
        batch: LMBatch,
    ):
        raise NotImplemented()

    def cuda(self):
        self.all_torch_models.cuda()

    def cpu(self):
        self.all_torch_models.cpu()

    def start_train_session(self):
        self.optimizer = Adam(self.all_torch_models.parameters())
        self.all_torch_models.train()

    def train_batch(
        self,
        batch: LMBatch
    ) -> Tuple[float, float, float]:
        self.optimizer.zero_grad()
        lm_predictions, next_sent_pred = self._predict(batch, for_loss_input=True)
        next_sent_loss = self._get_next_sentence_pred_loss(next_sent_pred, batch.is_sequential)
        mask_task_loss = self._get_mask_task_loss(lm_predictions, batch.mask_expected_ind)
        total_loss = next_sent_loss + mask_task_loss
        total_loss.backward()
        self.optimizer.step()
        if math.isnan(float(total_loss)):
            # This wierdly happened onece, so adding a checks and prints of debug info
            # for if happens again.
            print("\nUnexpected NAN???")
            print("ns loss", next_sent_loss)
            print("mask loss", mask_task_loss)
            print("batch", batch)
            print("lm_pred", lm_predictions)
            print("next_set_pred", next_sent_pred)
            raise ValueError("Unexpected NAN")
        return float(next_sent_loss), float(mask_task_loss), float(total_loss)

    def _get_next_sentence_pred_loss(self, pred_no_sigmoid, expected) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred_no_sigmoid, expected.float())

    def _get_mask_task_loss(self, pred_vals, expected_tok_inds) -> torch.Tensor:
        if len(pred_vals) == 0:
            return torch.tensor(0.0)
        flat_expected = torch.cat([v for v in expected_tok_inds if len(v) > 0])
        return F.cross_entropy(pred_vals, flat_expected)

    def _predict(self, batch: LMBatch, for_loss_input: bool = False):
        x = self.base_encoder(batch.tokens, batch.token_case_mod, batch.token_whitespace_mod,
                              batch.original_token_counts)
        return (
            self.lm_head(x, batch.mask_inds, not for_loss_input),
            self.next_sent_head(x, batch.original_token_counts, apply_sigmoid=not for_loss_input)
        )

    def get_save_state_dict(self):
        return {
            "version": 0,
            "name": "CookieMonsterForPretraining",
            "base_encoder": self.base_encoder,
            "head_states": self.heads.state_dict()
        }

    @classmethod
    def create_from_save_state_dict(
        cls,
        state_dict: dict,
    ) -> 'CookieMonsterForPretraining':
        instance = cls(
            state_dict['base_encoder']
        )
        instance.heads.load_state_dict(state_dict['head_states'])
        return instance


class CookieMonsterBaseEncoder(ModTokensEncoder):
    def __init__(
        self,
        base_embedder: Multiembedder,
        hidden_size_base: int = 128,
        num_layers: int = 2
    ):
        super().__init__()
        self.hidden_size_base = hidden_size_base
        self.embedder = base_embedder
        self.after_embed_dropout = nn.Dropout(p=0.1)
        self.conv1 = Conv1dSame(hidden_size_base, hidden_size_base, 3,
                                tokens_before_channels=True,
                                groups=4)
        self.rnn2 = RNNSeqEncoder(hidden_size_base, hidden_size_base, None, hidden_size_base,
                                  variable_lengths=True, num_layers=num_layers, input_dropout_p=0.1)
        self.transformer = ainix_kernel.model_util.transformer.layers.EncoderLayer(
            hidden_size=hidden_size_base,
            total_key_depth=hidden_size_base,
            total_value_depth=hidden_size_base,
            filter_size=hidden_size_base,
            num_heads=12,
            layer_dropout=0.1,
            relu_dropout=0.1
        )
        #self.rnn3 = RNNSeqEncoder(hidden_size_base, hidden_size_base, None, hidden_size_base,
        #                          variable_lengths=True)

    def forward(
        self,
        token_inds: torch.LongTensor,
        case_mod_inds: torch.LongTensor,
        whitespace_mod_inds: torch.LongTensor,
        input_lens: Optional[torch.LongTensor] = None
    ):
        embeded = self.embedder.embed(
            torch.stack((token_inds, case_mod_inds, whitespace_mod_inds)))
        start_shape = embeded.shape
        x = self.after_embed_dropout(embeded)
        x = self.conv1(x)
        x = self.rnn2(x, input_lens)
        x = self.transformer(x)
        new_blend_alpha = 1
        x = x*new_blend_alpha + embeded*(1-new_blend_alpha)
        assert x.shape[0] == start_shape[0] and x.shape[1] == start_shape[1]
        return x

    @property
    def output_size(self) -> int:
        return self.hidden_size_base

    def get_tokens_input_weights(self) -> torch.Tensor:
        """For the BERT task the prediction of the tokens shares the weights with
        base embedding. Get this weight"""
        return self.embedder.embedders[0].weight


class CookieMonsterLMPredictionHead(nn.Module):
    def __init__(self, model_embedding_weights):
        super().__init__()
        #self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(model_embedding_weights.size(1),
                                 model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(model_embedding_weights.size(0)))

    def forward(
        self,
        hidden_states,
        mask_inds,
        apply_softmax=True
    ):
        #hidden_states = self.transform(hidden_states)
        hidden_states = pack_picks(hidden_states, mask_inds)
        if len(hidden_states) == 0:
            return hidden_states
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class NextSentenceCookieMonsterHead(nn.Module):
    """Given some hidden states, predicts whether two sentece are seqentual"""
    def __init__(self, input_size):
        super().__init__()
        self.pooled_feed_forward = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            GELUActivation(),
            nn.Linear(input_size // 2, 1)
        )

    def forward(self, data: torch.Tensor, input_lens, apply_sigmoid = True):
        """

        Args:
            data: of dim (batch, seq_len, hidden_size)

        Returns:
            Tensor of dim (batch, ) predicting whether sequential. If
            apply_sigmoid is true, then this 0 to 1. Otherwise is a logit.
        """
        pooled = avg_pool(data, input_lens)
        pooled = self.pooled_feed_forward(pooled)
        pooled = pooled.squeeze(1)
        return pooled if not apply_sigmoid else torch.sigmoid(pooled)


class PretrainPoweredQueryEncoder(QueryEncoder):
    def __init__(
        self,
        tokenizer: ModifiedWordPieceTokenizer,
        query_vocab: Vocab,
        initial_encoder: ModTokensEncoder,
        summary_size: int,
        device: torch.device = torch.device("cpu"),
        learned_extra_transform: bool = False
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.query_vocab = query_vocab
        self.initial_encoder = initial_encoder
        self.device = device

        self.learend_extra_transform = learned_extra_transform
        other_modules = []
        if learned_extra_transform:
            self.pre_summary = nn.Sequential(
                nn.Dropout(p=0.15),
                nn.Linear(self.initial_encoder.output_size, summary_size),
            )
            self.post_summary_linear = nn.Linear(summary_size, summary_size)
            other_modules = [self.pre_summary, self.post_summary_linear]
        # To avoid storing redunant copies of the weights we store a list of
        # models that we need to save the weights of during serialization
        self.other_models = nn.ModuleList(other_modules)
        self.summary_size = summary_size

    def _vectorize_query(self, queries: Sequence[str]):
        """Converts a batch of string queries into dense vectors"""
        tokens, metads = zip(*self.tokenizer.tokenize_batch(queries))
        tok_inds, case, ws, origional_lens = torchify_batch_modded_tokens(
            tokens, self.query_vocab, self.device)
        hidden = self.initial_encoder(tok_inds, case, ws, origional_lens)
        return hidden, tokens, origional_lens, metads

    def forward(
        self,
        queries: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[ModifiedStringToken]]]:
        vectorized, tokenized, input_lens, token_metads = self._vectorize_query(queries)
        if self.learend_extra_transform:
            vectorized_for_summary = self.pre_summary(vectorized)
        summaries = self._sumarize(vectorized_for_summary, input_lens, tokenized, token_metads)
        return summaries, vectorized, tokenized

    @classmethod
    def sumarize(cls, hidden, input_lens, tokens, metads):
        stop_word_masks = get_non_stop_word_mask_batched(tokens, metads)
        # Average pool the tokens.
        # Count the tokens of long words less so that way a single long word like
        # a file name does not dominate the sumamry
        word_lens = get_word_lens_of_moded_tokens(tokens)
        stop_word_masks *= cls._word_lens_to_weights(word_lens)
        # Limit SOS or EOS
        stop_word_masks[:, 0] = 0.5
        stop_word_masks[:, -1] = 0.5
        #print(stop_word_masks)
        #
        weights = stop_word_masks.unsqueeze(2).expand(-1, -1, hidden.shape[-1])
        avgs = torch.sum((hidden * weights), dim=1) / torch.sum(weights, dim=1)
        #if self.learend_extra_transform:
        #    avgs = self.post_summary_linear(avgs)

        return avgs

    @classmethod
    def _word_lens_to_weights(cls, word_lens):
        originally_wanted_grad = word_lens.requires_grad  # for sanity check
        # Pretend that pads are really long
        word_lens = word_lens.clone()
        word_lens[word_lens == 0] = 100000
        # As words get longer, it should approach treating all the tokens in that
        # word as one token. However, if we have just two or so tokens in a word,
        # then we won't quite equally split them
        # equivalent_to_toks is a calculates how many tokens a len is worth.
        # unweighted a word with 5 len would count as 5 tokens. However, we are
        # setting it to a smaller value (~3 in this case).
        # The function used is somewhat arbitrary. One was just created that roughly
        # had the desired properties. It could probably be simplified
        #equivalent_to_toks = 1 + (word_lens - 1) * 0.4 + (1 - 1/word_lens)
        equivalent_to_toks = word_lens
        weights = 1.0 / equivalent_to_toks
        assert weights.requires_grad == originally_wanted_grad  # sanity check
        return weights


    def get_tokenizer(self) -> tokenizers.ModifiedWordPieceTokenizer:
        return self.tokenizer

    def get_save_state_dict(self):
        return {
            "version": 0,
            "tokenizer": self.tokenizer.get_save_state_dict(),
            "query_vocab": self.query_vocab.get_save_state_dict(),
            # TODO (DNGros): Make custom handling of serialize deserialize
            # feeling lazy right now so just going to pickle the internal encoder
            "initial_encoder": self.initial_encoder,
            "other_models_state": self.other_models.state_dict(),
            "summary_size": self.summary_size
        }

    @classmethod
    def create_from_save_state_dict(
        cls,
        state_dict: dict
    ) -> 'PretrainPoweredQueryEncoder':
        instance = cls(
            tokenizer=tokenizers.tokenizer_from_save_dict(state_dict['tokenizer']),
            query_vocab=BasicVocab.create_from_save_state_dict(state_dict['query_vocab']),
            initial_encoder=state_dict['initial_encoder'],
            summary_size=state_dict['summary_size']
        )
        instance.other_models.load_state_dict(state_dict['other_models_state'])
        return instance

    @classmethod
    def create_with_pretrained_checkpoint(
        cls,
        pretrained_checkpoint_path: str,
        tokenizer: ModifiedWordPieceTokenizer,
        query_vocab: Vocab,
        summary_size: int,
        device: torch.device = torch.device("cpu"),
        freeze_base = False
    ) -> 'PretrainPoweredQueryEncoder':
        """
        Creates a new instance using the base encoder from a encoder that was
        trained in the pretraining task.
        Args:
            pretrained_checkpoint_path: A path which when loaded gets the save
                state dict from a CookieMonsterForPretraining
            See __init__ for other args...
        """
        pretrainer_model = torch.load(pretrained_checkpoint_path, map_location=device)
        initial_encoder = pretrainer_model['model']['base_encoder'].to(device)
        if freeze_base:
            for param in initial_encoder.parameters():
                param.requires_grad = False
        return cls(
            tokenizer, query_vocab, initial_encoder, summary_size, device
        )


def make_default_cookie_monster_base(
    vocab: Vocab,
    hidden_size_base: int,
    num_layers: int = 2
):
    embedder = Multiembedder(
        (len(vocab), len(CasingModifier), len(WhitespaceModifier)),
        target_out_len=hidden_size_base
    )
    return CookieMonsterBaseEncoder(embedder, hidden_size_base, num_layers)


def make_default_cookie_monster(
    vocab: Vocab,
    hidden_size_base: int,
    use_cuda: bool
) -> BertlikeLangModel:
    return CookieMonsterForPretraining(
        make_default_cookie_monster_base(vocab, hidden_size_base), use_cuda)
