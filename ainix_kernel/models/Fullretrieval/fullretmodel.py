import collections
import math

import torch
from typing import Tuple, Optional, List

from tqdm import tqdm

from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode, CopyNode, \
    AstIterPointer
from ainix_common.parsing.copy_tools import make_copy_version_of_tree, get_paths_to_all_copies
from ainix_common.parsing.model_specific import tokenizers, parse_constants
from ainix_common.parsing.model_specific.tokenizers import get_default_pieced_tokenizer_word_list, \
    AstValTokenizer
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import ExamplesStore, Example, DataSplits
from ainix_kernel.model_util.attending import attend
from ainix_kernel.model_util.vocab import Vocab, BasicVocab
from ainix_kernel.models.EncoderDecoder.encdecmodel import make_default_query_encoder, \
    get_default_tokenizers
from ainix_kernel.models.EncoderDecoder.encoders import StringQueryEncoder
from ainix_kernel.models.model_types import StringTypeTranslateCF, TypeTranslatePredictMetadata
from ainix_kernel.training.augmenting.replacers import Replacer
import numpy as np
import torch.nn.functional as F

import attr

REPLACEMENT_SAMPLES = 30


class FullRetModel(StringTypeTranslateCF):
    def __init__(
        self,
        embedder: StringQueryEncoder,
        summaries: torch.Tensor,
        dataset_splits: torch.Tensor,
        example_refs: np.ndarray
    ):
        self.embedder = embedder
        self.summaries = summaries
        self.dataset_splits = dataset_splits
        self.example_refs = example_refs
        self.not_train_mask = self.dataset_splits != DataSplits.TRAIN.value

    def predict(
        self,
        x_string: str,
        y_type_name: str,
        use_only_train_data: bool
    ) -> Tuple[ObjectChoiceNode, 'TypeTranslatePredictMetadata']:
        self.embedder.eval()
        summary, memory, tokens = self.embedder([x_string])
        sims = F.cosine_similarity(summary, self.summaries)
        if use_only_train_data:
            sims[self.not_train_mask] = -1
        top_sims, top_inds = torch.topk(sims, 5)
        example_ref: _ExampleRef = self.example_refs[int(top_inds[0])]
        ref_ast = example_ref.reference_ast
        ast_with_new_copies = self._apply_copy_changes(
            ref_ast, example_ref.copy_refs, memory)
        metad = TypeTranslatePredictMetadata((math.log(float(top_sims[0])), ), None)
        return ast_with_new_copies, metad

    def _apply_copy_changes(
        self,
        ref_ast: ObjectChoiceNode,
        copy_refs: List['_CopyNodeReference'],
        embedded_tokens: torch.Tensor
    ) -> ObjectChoiceNode:
        latest_tree = ref_ast
        for copy_ref in copy_refs:
            copy_node_pointer = latest_tree.get_node_along_path(copy_ref.path_to_this_copy)
            new_copy_node = self._figure_out_new_copy_node(
                copy_ref, embedded_tokens, copy_node_pointer.cur_node.copy_type)
            latest_tree = copy_node_pointer.change_here(new_copy_node).get_root().cur_node
        return latest_tree

    def _figure_out_new_copy_node(
        self,
        copy_ref: '_CopyNodeReference',
        embedded_tokens: torch.Tensor,
        copy_type
    ) -> CopyNode:
        assert len(embedded_tokens) == 1, "no batch yet :("
        # TODO valid for copy mask
        start_attens = attend.get_attend_weights(
            copy_ref.start_atten_vec.unsqueeze(0).unsqueeze(0),
            embedded_tokens, normalize='identity')
        starts = torch.argmax(start_attens, dim=2)
        end_attens = attend.get_attend_weights(
            copy_ref.end_atten_vec.unsqueeze(0).unsqueeze(0),
            embedded_tokens, normalize='identity')
        ends = torch.argmax(end_attens, dim=2)
        return CopyNode(copy_type, int(starts[0, 0]), int(ends[0, 0]))

    def train(
        self,
        x_string: str,
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode,
        example_id: int
    ):
        raise NotImplemented("This thing doesn't need training")

    @classmethod
    def make_examples_store(
        cls,
        type_context: TypeContext,
        is_training: bool
    ) -> ExamplesStore:
        raise NotImplemented()

    def get_string_tokenizer(self) -> tokenizers.StringTokenizer:
        return self.embedder.get_tokenizer()


@attr.s(auto_attribs=True)
class _CopyNodeReference:
    path_to_this_copy: Tuple[int, ...]
    start_atten_vec: torch.Tensor
    end_atten_vec: torch.Tensor


@attr.s(auto_attribs=True)
class _ExampleRef:
    example_id: int
    reference_ast: ObjectChoiceNode
    copy_refs: Tuple[_CopyNodeReference, ...]


def _preproc_example(
    example: Example,
    replacers: Replacer,
    embedder: StringQueryEncoder,
    parser: StringParser,
    unparser: AstUnparser
) -> Tuple[torch.Tensor, _ExampleRef]:
    """Processedes an example for storing in the lookup model.

    Returns:k
        summary: A summary vector of this example
        ExampleRef: Metadata about this example that can be used when returning
            this in the model
    """
    # Sample a bunch of replacements and get their summary
    x, y = example.xquery, example.ytext
    needs_replacement = replacers.check_if_string_has_replacement_spots(x)
    xs, ys = [], []
    for _ in range(REPLACEMENT_SAMPLES if needs_replacement else 1):
        xreplaced, yreplaced = replacers.strings_replace(x, y)
        xs.append(xreplaced)
        ys.append(yreplaced)
    summaries, memories, tokens = embedder(xs)
    summary = torch.mean(summaries, dim=0)
    ast_parses = [parser.create_parse_tree(y, "CommandSequence") for y in ys]

    if needs_replacement:
        ast_as_copies = [
            make_copy_version_of_tree(
                ast, unparser,
                # Really should not have to retokenize, but whatever. It's memoized I think
                token_metadata=embedder.get_tokenizer().tokenize(rx)[1]
            )
            for ast, rx in zip(ast_parses, xs)
        ]
        paths_to_copies = [
            get_paths_to_all_copies(ast)
            for ast in ast_as_copies
        ]
        # With different replacements there are potentially different copy situations.
        # So let's take the most common one.
        # The 0th index of a call to most_common(1) is the most common. Then take [0] to get
        # actual path, not just the count of times it appears.
        most_common_paths = collections.Counter(paths_to_copies).most_common(1)[0][0]
        # Our representive ast will the ast we will return in this summary matches, with
        # just with all the copy nodes swapped out as appropriate
        for ast, paths in zip(ast_as_copies, paths_to_copies):
            if paths == most_common_paths:
                representive_ast = ast
                break
        else:
            raise RuntimeError("Should have found match?")
        # Pull out the average start and end vector for every copy in the representive_ast
        copy_refs = []
        for copy_path in most_common_paths:
            starts, ends = [], []
            for ast_with_copies, this_ast_paths, embedded_toks in zip(
                    ast_as_copies, paths_to_copies, memories):
                if this_ast_paths != most_common_paths:
                    continue
                actual_node = ast_with_copies.get_node_along_path(copy_path).cur_node
                assert isinstance(actual_node, CopyNode)
                starts.append(embedded_toks[actual_node.start])
                ends.append(embedded_toks[actual_node.end])
            copy_refs.append(_CopyNodeReference(
                path_to_this_copy=copy_path,
                start_atten_vec=torch.mean(torch.stack(starts), dim=0),
                end_atten_vec=torch.mean(torch.stack(ends), dim=0)
            ))
    else:
        copy_refs = tuple()
        representive_ast = ast_parses[0]

    return (
        summary,
        _ExampleRef(example.example_id, representive_ast, tuple(copy_refs))
    )


def full_ret_from_example_store(
    example_store: ExamplesStore,
    replacers: Replacer,
    pretrained_checkpoint: str
) -> FullRetModel:
    output_size = 200
    (x_tokenizer, query_vocab), y_tokenizer = get_default_tokenizers()
    embedder = make_default_query_encoder(x_tokenizer, query_vocab,
                                       output_size, pretrained_checkpoint)
    embedder.eval()
    parser = StringParser(example_store.type_context)
    unparser = AstUnparser(example_store.type_context, embedder.get_tokenizer())
    processed_x_raws = set()
    summaries, example_refs, example_splits = [], [], []
    for example in tqdm(list(example_store.get_all_examples())):
        if example.xquery in processed_x_raws:
            continue
        # Get the most prefered y text
        all_y_examples = example_store.get_examples_from_y_set(example.y_set_id)
        all_y_examples = [e for e in all_y_examples if e.xquery == example.xquery]
        all_y_examples.sort(key=lambda e: e.weight)
        highest_rated_version_of_this_example = all_y_examples[0]
        assert highest_rated_version_of_this_example.xquery == example.xquery
        assert highest_rated_version_of_this_example.split == example.split
        new_summary, new_example_ref = _preproc_example(
            highest_rated_version_of_this_example, replacers, embedder, parser, unparser)
        summaries.append(new_summary)
        example_refs.append(new_example_ref)
        example_splits.append(example.split)
    return FullRetModel(
        embedder=embedder,
        summaries=torch.stack(summaries),
        dataset_splits=torch.tensor(example_splits),
        example_refs=np.array(example_refs)
    )
