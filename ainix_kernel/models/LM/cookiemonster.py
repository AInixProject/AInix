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
from typing import Optional, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from ainix_common.parsing.model_specific.tokenizers import CasingModifier, WhitespaceModifier
from ainix_kernel.model_util.lm_task_processor.lm_set_process import LMBatch
from ainix_kernel.model_util.operations import pack_picks
from ainix_kernel.model_util.usefulmodules import Conv1dSame
from ainix_kernel.model_util.vocab import Vocab
from ainix_kernel.models.EncoderDecoder.encoders import RNNSeqEncoder, ModTokensEncoder
from ainix_kernel.models.model_types import BertlikeLangModel
from ainix_kernel.multiembedder.multiencoder import Multiembedder


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
        self.torch_models = nn.ModuleList([self.base_encoder, self.lm_head, self.next_sent_head])
        self.optimizer: Optional[torch.optim.Optimizer] = None
        if use_cuda:
            self.torch_models.cuda()
        self.use_cuda = use_cuda

    def eval_run(
        self,
        batch: LMBatch,
    ):
        raise NotImplemented()

    def start_train_session(self):
        self.optimizer = Adam(self.torch_models.parameters())
        self.torch_models.train()

    def train_batch(
            self,
            batch: LMBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.torch_models.zero_grad()
        lm_predictions, next_sent_pred = self._predict(batch, for_loss_input=True)
        next_sent_loss = self._get_next_sentence_pred_loss(next_sent_pred, batch.is_sequential)
        mask_task_loss = self._get_mask_task_loss(lm_predictions, batch.mask_expected_ind)
        total_loss = next_sent_loss + mask_task_loss
        total_loss.backward()
        self.optimizer.step()
        return next_sent_loss, mask_task_loss, total_loss

    def _get_next_sentence_pred_loss(self, pred_no_sigmoid, expected) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred_no_sigmoid, expected.float())

    def _get_mask_task_loss(self, pred_vals, expected_tok_inds) -> torch.Tensor:
        if len(pred_vals) == 0:
            return torch.tensor(0.0)
        flat_expected = torch.cat([v for v in expected_tok_inds if len(v) > 0])
        return F.cross_entropy(pred_vals, flat_expected)

    def _predict(self, batch: LMBatch, for_loss_input: bool = False):
        x = self.base_encoder(batch.tokens, batch.token_case_mod, batch.token_whitespace_mod)
        return (
            self.lm_head(x, batch.mask_inds, not for_loss_input),
            self.next_sent_head(x, apply_sigmoid=not for_loss_input)
        )

    def get_save_state_dict(self):
        return self.torch_models.state_dict()


class CookieMonsterBaseEncoder(ModTokensEncoder):
    def __init__(
        self,
        base_embedder: Multiembedder,
        hidden_size_base: int = 128
    ):
        super().__init__()
        self.hidden_size_base = hidden_size_base
        self.embedder = base_embedder
        self.conv1 = Conv1dSame(hidden_size_base, hidden_size_base, 3, tokens_before_channels=True)
        self.rnn2 = RNNSeqEncoder(hidden_size_base, hidden_size_base, None, hidden_size_base)
        self.rnn3 = RNNSeqEncoder(hidden_size_base, hidden_size_base, None, hidden_size_base)
        self.torch_models = nn.ModuleList([
            self.embedder, self.conv1, self.rnn2, self.rnn3])

    def forward(
        self,
        token_inds: torch.LongTensor,
        case_mod_inds: torch.LongTensor,
        whitespace_mod_inds: torch.LongTensor
    ):
        x = self.embedder.embed(
            torch.stack((token_inds, case_mod_inds, whitespace_mod_inds)))
        start_shape = x.shape
        x = self.conv1(x)
        x = self.rnn2(x)
        x = self.rnn3(x)
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
            nn.ReLU(),
            nn.Linear(input_size // 2, 1)
        )

    def forward(self, data: torch.Tensor, apply_sigmoid = True):
        """

        Args:
            data: of dim (batch, seq_len, hidden_size)

        Returns:
            Tensor of dim (batch, ) predicting whether sequential. If
            apply_sigmoid is true, then this 0 to 1. Otherwise is a logit.
        """
        pooled = torch.sum(data, dim=1) / float(data.shape[1])
        pooled = self.pooled_feed_forward(pooled)
        pooled = pooled.squeeze(1)
        return pooled if not apply_sigmoid else torch.sigmoid(pooled)




def make_default_cookie_monster(
    vocab: Vocab,
    hidden_size_base: int,
    use_cuda: bool
) -> BertlikeLangModel:
    embedder = Multiembedder(
        (len(vocab), len(CasingModifier), len(WhitespaceModifier)),
        target_out_len=hidden_size_base
    )
    encoder = CookieMonsterBaseEncoder(embedder, hidden_size_base)
    return CookieMonsterForPretraining(encoder, use_cuda)
