from argparse import ArgumentParser
from typing import List, Generator, Iterable, Tuple, Set, Callable

from ainix_common.parsing.ast_components import AstSet, AstObjectChoiceSet
from ainix_common.parsing.model_specific.tokenizers import StringTokenizer, get_tokenizer_by_name, \
    nonascii_untokenize
from ainix_common.parsing.parse_primitives import AInixParseError
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import ExamplesStore
from ainix_kernel.models.model_types import ModelCantPredictException, ModelSafePredictError
from ainix_kernel.training.augmenting.replacers import ReplacementSampling
from ainix_kernel.training.evaluate import EvaluateLogger, AstEvaluation, print_ast_eval_log
from ainix_kernel.training.trainer import get_examples, make_y_ast_set


def get_y_ast_sets(
    xs: List[str],
    yids: List[int],
    yhashes: List[str],
    rsamples: List[ReplacementSampling],
    string_tokenizer: StringTokenizer,
    index: ExamplesStore
) -> Generator[Tuple[AstSet, Set[str]], None, None]:
    type_context = index.type_context
    string_parser = StringParser(type_context)
    unparser = AstUnparser(type_context, string_tokenizer)
    verify_same_hashes(index, yids, yhashes)
    assert len(xs) == len(yids) == len(rsamples)
    for x, yid, rsample in zip(xs, yids, rsamples):
        yvalues = index.get_y_values_for_y_set(yid)
        this_x_toks, this_x_metadata = string_tokenizer.tokenize(x)
        y_ast_set, y_texts, teacher_force_path_ast = make_y_ast_set(
            y_type=type_context.get_type_by_name(yvalues[0].y_type),
            all_y_examples=yvalues,
            replacement_sample=rsample,
            string_parser=string_parser,
            this_x_metadata=this_x_metadata,
            unparser=unparser
        )
        yield y_ast_set, y_texts


def eval_stuff(
    xs: List[str],
    ys: List[str],
    preds: List[str],
    y_ast_sets: Iterable[Tuple[AstObjectChoiceSet, Set[str]]],
    string_parser: StringParser,
    unparser: AstUnparser
):
    logger = EvaluateLogger()
    assert len(xs) == len(preds)
    for x, pred, (y_ast_set, y_texts) in zip(xs, preds, y_ast_sets):
        pexception = None
        pred_ast = None
        try:
            pred_ast = string_parser.create_parse_tree(pred, y_ast_set.type_to_choose.name)
        except AInixParseError as e:
            pexception = e
        evaluation = AstEvaluation(pred_ast, y_ast_set, y_texts, x,
                                   pexception, unparser, pred)
        evaluation.print_vals(unparser)
        logger.add_evaluation(evaluation)
    print_ast_eval_log(logger)


def verify_same_hashes(index: ExamplesStore, yids, yhashes):
    for yid, yhash in zip(yids, yhashes):
        if index.get_y_set_hash(yid) != yhash:
            for yv in index.get_y_values_for_y_set(yid):
                print(yv)
            print(index.get_y_set_hash(yid))
            raise ValueError("Not matching hash")


if __name__ == "__main__":
    argparer = ArgumentParser()
    argparer.add_argument("--src_xs", type=str)
    argparer.add_argument("--predictions", type=str)
    argparer.add_argument("--tgt_ys", type=str)
    argparer.add_argument("--tgt_yids", type=str)
    argparer.add_argument("--tokenizer_name", type=str)
    args = argparer.parse_args()

    with open(args.src_xs, 'r') as f:
        xs = f.readlines()
    with open(args.tgt_ys, 'r') as f:
        ys = f.readlines()
    with open(args.predictions, 'r') as f:
        predictions = f.readlines()
    with open(args.tgt_yids, 'r') as f:
        yinfo = f.readlines()
    xs = list(map(nonascii_untokenize, xs))
    predictions = list(map(nonascii_untokenize, predictions))
    yids = []
    yhashes = []
    rsamples = []
    for info in yinfo:
        split_info = info.split()
        yids.append(int(split_info[0]))
        yhashes.append(split_info[1])
        rsamples.append(ReplacementSampling.from_serialized_string(
            "".join(split_info[2:])
        ))
    tokenizer = get_tokenizer_by_name(args.tokenizer_name)
    type_context, index, replacers, loader = get_examples()
    string_parser = StringParser(type_context)
    unparser = AstUnparser(type_context, tokenizer)
    y_asts = get_y_ast_sets(
        xs, yids, yhashes, rsamples, tokenizer, index
    )
    eval_stuff(xs, ys, predictions, y_asts, string_parser, unparser)