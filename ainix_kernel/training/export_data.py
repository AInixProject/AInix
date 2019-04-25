import itertools
from argparse import ArgumentParser
from collections import defaultdict

from ainix_common.parsing.model_specific.tokenizers import get_default_pieced_tokenizer_word_list, \
    NonLetterTokenizer
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_kernel.indexing.examplestore import DataSplits, DEFAULT_SPLITS
from ainix_kernel.training.trainer import iterate_data_pairs, get_examples
from tqdm import tqdm

if __name__ == "__main__":
    argparer = ArgumentParser()
    argparer.add_argument("--prefix", type=str, default="data_")
    argparer.add_argument("--replace_samples", type=int, default=1)
    default_split_train = DEFAULT_SPLITS[0]
    assert default_split_train[1] == DataSplits.TRAIN
    argparer.add_argument("--train_percent", type=float, default=default_split_train[0]*100)
    argparer.add_argument("--randomize_seed", type=bool, default=False)
    args = argparer.parse_args()

    train_frac = args.train_percent / 100.0
    split_proportions = ((train_frac, DataSplits.TRAIN), (1-train_frac, DataSplits.VALIDATION))

    type_context, index, replacers, loader = get_examples(
        split_proportions, randomize_seed=args.randomize_seed)
    index.get_all_examples(())
    string_parser = StringParser(type_context)
    tokenizer, vocab = get_default_pieced_tokenizer_word_list()
    unparser = AstUnparser(type_context, tokenizer)
    non_ascii_tokenizer = NonLetterTokenizer()
    def non_asci_do(string):
        toks, metad = non_ascii_tokenizer.tokenize(string)
        return " ".join(toks)

    split_to_sentences = defaultdict(list)
    num_to_do = args.replace_samples*index.get_doc_count()
    data_iterator = itertools.chain.from_iterable(
        (iterate_data_pairs(
            index, replacers, string_parser, tokenizer, unparser, None)
         for epoch in range(args.replace_samples))
    )
    data_iterator = itertools.islice(data_iterator, num_to_do)
    for (example, this_example_replaced_x, y_ast_set,
         teacher_force_path_ast, y_texts, rsample) in tqdm(data_iterator, total=num_to_do):
            teach_force_str = unparser.to_string(
                teacher_force_path_ast, this_example_replaced_x).total_string
            split_to_sentences[example.split].append((
                non_asci_do(this_example_replaced_x),
                non_asci_do(teach_force_str),
                example.y_set_id,
                rsample
             ))
    for split, datas in split_to_sentences.items():
        split = {0: 'train', 1: 'val'}[split]
        with open(f'{args.prefix}{split}_x.txt', 'w') as xf, \
                open(f'{args.prefix}{split}_y.txt', 'w') as yf, \
                open(f'{args.prefix}{split}_yids.txt', 'w') as yidsf:
            xs, ys, y_set_ids, rsamples = zip(*datas)
            xf.write("\n".join(xs))
            yf.write("\n".join(ys))
            #if split == "val":
            #    for y_set_id in y_set_ids:
            #        print(index.get_y_values_for_y_set(y_set_id))
            yidsf.write("\n".join([
                f"{y_set_id} {index.get_y_set_hash(y_set_id)} {rs.serialize_to_string()}"
                for y_set_id, rs in zip(y_set_ids, rsamples)
            ]))
