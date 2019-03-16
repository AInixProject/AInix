import itertools
from collections import defaultdict

from ainix_common.parsing.model_specific.tokenizers import get_default_pieced_tokenizer_word_list, \
    NonLetterTokenizer
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_kernel.training.fullret_try import get_examples
from ainix_kernel.training.trainer import iterate_data_pairs
from tqdm import tqdm

# TODO move these constants to args
NUM_REPLACER_SAMPLES = 10
PREFIX = "data_"

if __name__ == "__main__":
    type_context, index, replacers = get_examples()
    index.get_all_examples(())
    string_parser = StringParser(type_context)
    tokenizer, vocab = get_default_pieced_tokenizer_word_list()
    unparser = AstUnparser(type_context, tokenizer)
    non_ascii_tokenizer = NonLetterTokenizer()
    def non_asci_do(string):
        toks, metad = non_ascii_tokenizer.tokenize(string)
        return " ".join(toks)

    split_to_sentences = defaultdict(list)
    num_to_do = NUM_REPLACER_SAMPLES*index.get_doc_count()
    data_iterator = itertools.chain.from_iterable(
        (iterate_data_pairs(
            index, replacers, string_parser, tokenizer, unparser, None)
         for epoch in range(NUM_REPLACER_SAMPLES))
    )
    data_iterator = itertools.islice(data_iterator, num_to_do)
    for (example, this_example_replaced_x, y_ast_set,
         teacher_force_path_ast, y_texts) in tqdm(data_iterator, total=num_to_do):
            teach_force_str = unparser.to_string(
                teacher_force_path_ast, this_example_replaced_x).total_string
            split_to_sentences[example.split].append((
                non_asci_do(this_example_replaced_x),
                non_asci_do(teach_force_str)
             ))
    for split, datas in split_to_sentences.items():
        split = {0: 'train', 1: 'val'}[split]
        with open(f'{PREFIX}{split}_x.txt', 'w') as xf, open(f'{PREFIX}{split}_y.txt', 'w') as yf:
            xs, ys = zip(*datas)
            xf.write("\n".join(xs))
            yf.write("\n".join(ys))
