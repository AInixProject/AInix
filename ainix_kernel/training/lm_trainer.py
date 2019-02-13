import collections

from ainix_common.parsing.model_specific import tokenizers, parse_constants
from ainix_kernel.model_util.lm_task_processor.lm_set_process import CookieMonsterDataset, \
    CookieMonsterBatchIterator
from ainix_kernel.model_util.vocab import BasicVocab
from ainix_kernel.models.LM.cookiemonster import CookieMonster, make_default_cookie_monster
from ainix_kernel.models.model_types import BertlikeLangModel

class MovingAverage:
    def __init__(self, window: int):
        self.vals = collections.deque(maxlen=window)
        self.cum_sum = 0

    def add(self, val):
        self.cum_sum += val
        if len(self.vals) == self.vals.maxlen:
            self.cum_sum -= self.vals[0]
        self.vals.append(val)

    def get(self):
        return self.cum_sum / len(self.vals)

class BertlikeTrainer:
    def __init__(
        self,
        model: BertlikeLangModel,
        dataset: CookieMonsterDataset,
        batch_size: int = 1
    ):
        self.model = model
        self.dataset = dataset
        self.batch_iter = CookieMonsterBatchIterator(dataset, batch_size)

    def train(self, iterations: int):
        self.loss_avg = MovingAverage(10)
        for iteration in range(iterations):
            batch = next(self.batch_iter)
            loss = self.model.train_batch(batch)
            self.loss_avg.add(loss)
            print(f"Iter {iteration} loss {self.loss_avg.get()}")


if __name__ == "__main__":
    tokenizer, vocab_list = tokenizers.get_default_pieced_tokenizer_word_list()
    vocab = BasicVocab(vocab_list + parse_constants.ALL_SPECIALS)
    dataset = CookieMonsterDataset(
        ["../../builtin_types/otherdata/stackexchange/unix-stackexchange/sentences.txt"],
        tokenizer, vocab
    )
    model = make_default_cookie_monster(vocab, hidden_size_base=64)
    batch_size = 32
    trainer = BertlikeTrainer(model, dataset, batch_size=batch_size)
    trainer.train(int(1e5 / batch_size))
