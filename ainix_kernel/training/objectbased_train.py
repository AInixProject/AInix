import itertools
from argparse import ArgumentParser

from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_kernel.indexing.modelbasedstores import cache_examples_from_iterable, EncoderCache
from ainix_kernel.models.EncoderDecoder.encdecmodel import make_default_query_encoder, \
    get_default_tokenizers
from ainix_kernel.training.trainer import get_examples, iterate_data_pairs
from tqdm import tqdm

DEFAULT_REPLACE_SAMPLES = 5
pretrain_checkpoint="../../checkpoints/" \
                    "lmchkp_30epoch2rnn_merge_toks_total_2.922_ns0.424_lm2.4973.pt"
DEFAULT_ENCODER_BATCH_SIZE = 1


def do_train(
    num_replace_samples = DEFAULT_REPLACE_SAMPLES
):
    type_context, index, replacers, loader = get_examples()
    (x_tokenizer, x_vocab), y_tokenizer = get_default_tokenizers()

    string_parser = StringParser(type_context)
    unparser = AstUnparser(type_context, x_tokenizer)

    rsampled_examples = tqdm(itertools.chain.from_iterable((
        iterate_data_pairs(index, replacers, string_parser, x_tokenizer, unparser, None)
        for _ in range(num_replace_samples)
    )), total=index.get_num_x_values() * num_replace_samples)

    # TODO this should probably be model dependent
    encoder = make_default_query_encoder(x_tokenizer, x_vocab, 200, pretrain_checkpoint)

    encoder_cache = EncoderCache(index)
    cache_examples_from_iterable(
        encoder_cache, encoder, rsampled_examples, DEFAULT_ENCODER_BATCH_SIZE)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-rsamples", default=DEFAULT_REPLACE_SAMPLES)
    args = argparser.parse_args()
    do_train(args.rsamples)
