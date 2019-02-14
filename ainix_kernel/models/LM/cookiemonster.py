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
from ainix_kernel.models.EncoderDecoder.encoders import RNNSeqEncoder
from ainix_kernel.models.model_types import BertlikeLangModel
from ainix_kernel.multiembedder.multiencoder import Multiembedder


class CookieMonster(BertlikeLangModel):
    def __init__(
        self,
        base_embedder: Multiembedder,
        hidden_size_base: int = 128,
        use_cuda: bool = False
    ):
        self.embedder = base_embedder
        self.conv1 = Conv1dSame(hidden_size_base, hidden_size_base, 3, tokens_before_channels=True)
        self.rnn2 = RNNSeqEncoder(hidden_size_base, hidden_size_base, None, hidden_size_base)
        self.rnn3 = RNNSeqEncoder(hidden_size_base, hidden_size_base, None, hidden_size_base)
        self.next_sent_predictor = nn.Sequential(
            nn.Linear(hidden_size_base, hidden_size_base // 2),
            nn.ReLU(),
            nn.Linear(hidden_size_base // 2, 1)
        )
        self.lm_head = CookieMonsterLMPredictionHead(self.embedder.embedders[0].weight)
        #self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
        #    hidden_size_base, self.embedder.vocab_sizes[0], [10, 100, 1000])
        self.torch_models = nn.ModuleList([
            self.embedder, self.conv1, self.next_sent_predictor, self.rnn2,
            self.lm_head, self.rnn3])
        self.optimizer: Optional[torch.optim.Optimizer] = None
        if use_cuda:
            self.torch_models.cuda()
        self.use_cuda = use_cuda

    def eval_run(
        self,
        batch: LMBatch,
    ):
        pass

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
        flat_expected = torch.cat([v for v in expected_tok_inds if len(v) > 0])
        return F.cross_entropy(pred_vals, flat_expected)

    def _predict(self, batch: LMBatch, for_loss_input = False):
        x = self.embedder.embed(
            torch.stack((batch.tokens, batch.token_case_mod, batch.token_whitespace_mod)))
        start_shape = x.shape
        x = self.conv1(x)
        x = self.rnn2(x)
        x = self.rnn3(x)
        assert x.shape[0] == start_shape[0] and x.shape[1] == start_shape[1]
        return (
            self.lm_head(x, batch.mask_inds, not for_loss_input),
            self._predict_next_sentence(x, apply_sigmoid=not for_loss_input)
        )

    def _predict_next_sentence(self, data: torch.Tensor, apply_sigmoid = True):
        """

        Args:
            data: of dim (batch, seq_len, hidden_size)

        Returns:

        """
        flattened = torch.sum(data, dim=1) / float(data.shape[1])
        prediction = self.next_sent_predictor(flattened)
        prediction = prediction.squeeze(1)
        return prediction if not apply_sigmoid else torch.sigmoid(prediction)

    def get_save_state_dict(self):
        return self.torch_models.state_dict()


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
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


def make_default_cookie_monster(
    vocab: Vocab,
    hidden_size_base: int,
    use_cuda: bool
) -> BertlikeLangModel:
    embedder = Multiembedder(
        (len(vocab), len(CasingModifier), len(WhitespaceModifier)),
        target_out_len=hidden_size_base
    )
    return CookieMonster(embedder, hidden_size_base, use_cuda)
