from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from ainix_common.parsing.model_specific.tokenizers import CasingModifier, WhitespaceModifier
from ainix_kernel.model_util.lm_task_processor.lm_set_process import LMBatch
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
        self.conv1 = nn.Conv1d(hidden_size_base, hidden_size_base, 3)
        self.rnn2 = RNNSeqEncoder(hidden_size_base, hidden_size_base, None, hidden_size_base)
        self.next_sent_predictor = nn.Sequential(
            nn.Linear(hidden_size_base, hidden_size_base // 2),
            nn.ReLU(),
            nn.Linear(hidden_size_base // 2, 1)
        )
        self.tok_predict_linear = nn.Linear(hidden_size_base, hidden_size_base)
        self.torch_models = nn.ModuleList([
            self.embedder, self.conv1, self.next_sent_predictor, self.rnn2,
            self.tok_predict_linear])
        self.optimizer: Optional[torch.optim.Optimizer] = None
        if use_cuda:
            self.torch_models.cuda()
        self.use_cuda = use_cuda
        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            hidden_size_base, len(self.embedder.vocab_sizes[0]), [10, 100, 1000])

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
    ) -> torch.Tensor:
        self.torch_models.zero_grad()
        lm_predictions, next_sent_pred = self._predict(batch, for_loss_input=True)
        next_sent_loss = self._get_next_sentence_pred_loss(next_sent_pred, batch.is_sequential)
        mask_task_loss = self._get_mask_task_loss(lm_predictions, batch.mask_inds)
        next_sent_loss.backward()
        self.optimizer.step()
        return next_sent_loss

    def _get_next_sentence_pred_loss(self, pred_no_sigmoid, expected) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred_no_sigmoid, expected.float())

    def _get_mask_task_loss(self, pred_vals, expected_tok_inds) -> torch.Tensor:
        pred_vals =
        return self.adaptive_softmax(pred_vals)

    def _predict(self, batch: LMBatch, for_loss_input = False):
        x = self.embedder.embed(
            torch.stack((batch.tokens, batch.token_case_mod, batch.token_whitespace_mod)))
        x = x.transpose(1, 2)  # B x T x C -> B x C x T
        x = self.conv1(x)
        x = x.transpose(1, 2)  # B x C x T -> B x T x C
        x = self.rnn2(x)
        return (
            self._predict_masks(x, batch.mask_inds, not for_loss_input),
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

    def _predict_masks(self, data: torch.Tensor, mask_inds: List[torch.LongTensor], apply_softmax=True):
        x = _pack_picks(data, mask_inds)
        x = self.tok_predict_linear(x)
        x =
        return F.softmax(x) if apply_softmax else x


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
