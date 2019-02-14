import collections
import torch

from ainix_common.parsing.model_specific import tokenizers, parse_constants
from ainix_kernel.model_util.lm_task_processor.lm_set_process import CookieMonsterDataset, \
    CookieMonsterBatchIterator
from ainix_kernel.model_util.vocab import BasicVocab
from ainix_kernel.models.LM.cookiemonster import CookieMonster, make_default_cookie_monster
from ainix_kernel.models.model_types import BertlikeLangModel
from tqdm import tqdm


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
        batch_size: int = 1,
        serialize_callback = None
    ):
        self.model = model
        self.dataset = dataset
        self.batch_iter = CookieMonsterBatchIterator(dataset, batch_size)
        self.serialize_callback = serialize_callback

    def train(self, iterations: int, intermitted_save_path="./checkpoints/chkp"):
        self.model.start_train_session()
        window = 50
        self.total_loss_avg = MovingAverage(window)
        self.next_sent_loss_avg = MovingAverage(window)
        self.lm_loss_avg = MovingAverage(window)
        for iteration in tqdm(range(iterations)):
            batch = next(self.batch_iter)
            next_sent_loss, lm_loss, total_loss = self.model.train_batch(batch)
            self.next_sent_loss_avg.add(next_sent_loss)
            self.lm_loss_avg.add(lm_loss)
            self.total_loss_avg.add(total_loss)
            if iteration % window == 0:
                print(f"Iter {iteration} total loss {self.total_loss_avg.get()}. "
                      f"Next Sent {self.next_sent_loss_avg.get()}. LM {self.lm_loss_avg.get()}")
            if iteration % (window*20) == 0 and intermitted_save_path:
                s_path = f"{intermitted_save_path}_iter{iteration}_total_" + \
                         f"{total_loss}.pt"
                print(f"serializing to {s_path}")
                serialize_func(s_path, iteration*self.batch_iter.batch_size)


if __name__ == "__main__":
    tokenizer, vocab_list = tokenizers.get_default_pieced_tokenizer_word_list()
    use_cuda = True
    vocab = BasicVocab(vocab_list + parse_constants.ALL_SPECIALS,
                       default_device=torch.device("cuda" if use_cuda else "cpu"))
    dataset = CookieMonsterDataset(
        ["../../builtin_types/otherdata/stackexchange/unix-stackexchange/sentences.txt"],
        tokenizer, vocab, max_docs_to_load=100, use_cuda=use_cuda
    )
    model = make_default_cookie_monster(vocab, hidden_size_base=64, use_cuda=use_cuda)

    def serialize_func(path, seen_instances):
        ser = {
            "name": "CookieMonster",
            "version": 0,
            "model": model.get_save_state_dict(),
            "seen_instances": seen_instances
        }
        torch.save(ser, path)
    batch_size = 64
    trainer = BertlikeTrainer(model, dataset, batch_size=batch_size,
                              serialize_callback=serialize_func)
    trainer.train(int(1e5 / batch_size))
    serialize_func("saved_model.pt", 1e5)
