import collections
import math

import torch
from typing import Tuple, Optional, List, Dict, Callable

from tqdm import tqdm

from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode, CopyNode, \
    AstIterPointer
from ainix_common.parsing.copy_tools import make_copy_version_of_tree, get_paths_to_all_copies
from ainix_common.parsing.model_specific import tokenizers, parse_constants
from ainix_common.parsing.model_specific.tokenizers import get_default_pieced_tokenizer_word_list, \
    AstValTokenizer, ModifiedStringToken, WhitespaceModifier
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_common.parsing.typecontext import TypeContext, AInixType
from ainix_kernel.indexing.examplestore import ExamplesStore, XValue, DataSplits, YValue
from ainix_kernel.model_util.attending import attend
from ainix_kernel.model_util.operations import get_kernel_around
from ainix_kernel.model_util.vocab import Vocab, BasicVocab
from ainix_kernel.models.EncoderDecoder.decoders import get_valid_for_copy_mask
from ainix_kernel.models.EncoderDecoder.encdecmodel import make_default_query_encoder, \
    get_default_tokenizers
from ainix_kernel.models.EncoderDecoder.encoders import StringQueryEncoder
from ainix_kernel.models.model_types import StringTypeTranslateCF, TypeTranslatePredictMetadata, \
    ExampleRetrieveExplanation
from ainix_kernel.training.augmenting.replacers import Replacer
import numpy as np
import torch.nn.functional as F
import sklearn

import attr

REPLACEMENT_SAMPLES = 1
START_COPY_KERNEL_WEIGHTS = torch.tensor([0.25, 1, 0.05])
END_COPY_KERNEL_WEIGHTS = torch.tensor([0.05, 1, 0.25])
#                         ^ Weight current token the most and the before and after less.
COPY_KERNEL_SIZE = len(START_COPY_KERNEL_WEIGHTS)


class FullRetModel(StringTypeTranslateCF):
    def __init__(
        self,
        embedder: StringQueryEncoder,
        summaries: torch.Tensor,
        dataset_splits: torch.Tensor,
        example_refs: np.ndarray,
        nb_models: Dict[str, 'NBObjectChoiceModel']
    ):
        self.embedder = embedder
        self.summaries = summaries
        assert not self.summaries.requires_grad
        self.dataset_splits = dataset_splits
        self.example_refs = example_refs
        self.not_train_mask = self.dataset_splits != DataSplits.TRAIN.value
        self.nb_models = nb_models

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
        valid_for_copy_mask = get_valid_for_copy_mask(tokens)
        ast_with_new_copies = self._apply_copy_changes(
            ref_ast, example_ref.copy_refs, memory, valid_for_copy_mask, tokens[0])
        metad = TypeTranslatePredictMetadata(
            (math.log(float(top_sims[0])), ),
            (ExampleRetrieveExplanation(
                tuple([self.example_refs[int(ind)].x_val_id for ind in top_inds]),
                tuple([math.log(float(sim)) for sim in top_sims]),
                None),
            )
        )
        return ast_with_new_copies, metad

    def _apply_copy_changes(
        self,
        ref_ast: ObjectChoiceNode,
        copy_refs: List['_CopyNodeReference'],
        embedded_tokens: torch.Tensor,
        valid_for_copy_mask,
        original_tokens
    ) -> ObjectChoiceNode:
        latest_tree = ref_ast
        extra_feat_info = make_extra_feats_info(embedded_tokens[0], original_tokens)
        for copy_ref in copy_refs:
            copy_node_pointer = latest_tree.get_node_along_path(copy_ref.path_to_this_copy)
            new_copy_node = self._figure_out_new_copy_node(
                copy_ref, embedded_tokens, copy_node_pointer.cur_node.copy_type, extra_feat_info,
                valid_for_copy_mask)
            latest_tree = copy_node_pointer.change_here(new_copy_node).get_root().cur_node
            extra_feat_info = update_extra_feats_info(
                extra_feat_info, new_copy_node.start, new_copy_node.end)
        return latest_tree

    def _figure_out_new_copy_node(
        self,
        copy_ref: '_CopyNodeReference',
        embedded_tokens: torch.Tensor,
        copy_type,
        extra_copy_info,
        valid_for_copy_mask: torch.Tensor
    ) -> CopyNode:
        assert len(embedded_tokens) == 1, "no batch yet :("
        mask_addition = (1.0 - valid_for_copy_mask.float()) * -10000.0

        extra_vals = torch.cat(
            (embedded_tokens[0], get_extra_feat_vec(embedded_tokens[0], extra_copy_info)), dim=-1)
        extra_vals = extra_vals.unsqueeze(0)
        extra_valsBCT = extra_vals.transpose(1, 2)
        start_attens = self._apply_kernel(extra_valsBCT, copy_ref.start_avgs,
                                          START_COPY_KERNEL_WEIGHTS)
        start_attens += mask_addition
        assert start_attens.shape[1] == embedded_tokens.shape[1]
        starts = torch.argmax(start_attens, dim=1)
        end_attens = self._apply_kernel(extra_valsBCT, copy_ref.end_avgs, END_COPY_KERNEL_WEIGHTS)
        end_attens += mask_addition
        ends = torch.argmax(end_attens, dim=1)
        return CopyNode(copy_type, int(starts[0]), int(ends[0]))

    def _apply_kernel(self, valsBCT, kernel, k_weight):
        return F.conv1d(valsBCT, kernel.unsqueeze(0) * k_weight,
                        padding=COPY_KERNEL_SIZE // 2).squeeze(1)

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
    # Avgs are in shape (hidden_channel, kernel_size)
    # This is so it can more easily go into F.conv1d
    start_avgs: torch.Tensor
    end_avgs: torch.Tensor


@attr.s(auto_attribs=True)
class _ExampleRef:
    x_val_id: int
    y_val_id: int
    reference_ast: ObjectChoiceNode
    copy_refs: Tuple[_CopyNodeReference, ...]


def _preproc_example(
    xval: XValue,
    yval: YValue,
    replacers: Replacer,
    embedder: StringQueryEncoder,
    parser: StringParser,
    unparser: AstUnparser,
    nb_update_fn: Callable
) -> Tuple[torch.Tensor, _ExampleRef]:
    """Processedes an example for storing in the lookup model.

    Returns:k
        summary: A summary vector of this example
        ExampleRef: Metadata about this example that can be used when returning
            this in the model
    """
    # Sample a bunch of replacements and get their summary
    x, y = xval.x_text, yval.y_text
    needs_replacement = replacers.check_if_string_has_replacement_spots(x) or \
                        replacers.check_if_string_has_replacement_spots(y)
    xs, ys = [], []
    for _ in range(REPLACEMENT_SAMPLES if needs_replacement else 1):
        xreplaced, yreplaced = replacers.strings_replace(x, y)
        xs.append(xreplaced)
        ys.append(yreplaced)
    summaries, memories, tokens = embedder(xs)
    summary = torch.mean(summaries, dim=0)
    ast_parses = [parser.create_parse_tree(y, "CommandSequence") for y in ys]

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
    all_extra_feats = [make_extra_feats_info(m, toks) for m, toks in zip(memories, tokens)]
    for copy_path in most_common_paths:
        starts, ends = [], []
        for i, (ast_with_copies, this_ast_paths, embedded_toks, extra_feats) in \
                enumerate(zip(ast_as_copies, paths_to_copies, memories, all_extra_feats)):
            if this_ast_paths != most_common_paths:
                continue
            actual_node = ast_with_copies.get_node_along_path(copy_path).cur_node
            assert isinstance(actual_node, CopyNode)
            embedded_with_extra = torch.cat(
                (embedded_toks, get_extra_feat_vec(embedded_toks, extra_feats)),
                dim=-1)
            # TODO: Ahhh, should avoid repeatedly transposing
            embedded_with_extraBCT = embedded_with_extra.transpose(0, 1).unsqueeze(0)
            starts.append(
                get_kernel_around(embedded_with_extraBCT, actual_node.start).squeeze())
            ends.append(get_kernel_around(embedded_with_extraBCT, actual_node.end).squeeze())
            all_extra_feats[i] = update_extra_feats_info(
                extra_feats, actual_node.start, actual_node.end)
        copy_refs.append(_CopyNodeReference(
            path_to_this_copy=copy_path,
            start_avgs=torch.mean(torch.stack(starts), dim=0),
            end_avgs=torch.mean(torch.stack(ends), dim=0)
        ))

    if nb_update_fn:
        nb_update_fn(summary, representive_ast)

    return (
        summary,
        _ExampleRef(xval.id, yval.id, representive_ast, tuple(copy_refs))
    )


def make_extra_feats_info(curr_embed, original_tokens: List[ModifiedStringToken]):
    unpadded_len = len(original_tokens)
    return (
        torch.zeros((curr_embed.shape[0])),
        torch.zeros((curr_embed.shape[0])),
        torch.zeros((curr_embed.shape[0])),
        torch.tensor([1. if i < unpadded_len and
                            original_tokens[i].whitespace_modifier ==
                            WhitespaceModifier.AFTER_SPACE_OR_SOS else 0.
                     for i in range(curr_embed.shape[0])]),
        torch.tensor([1. if i == unpadded_len - 1 or (i < unpadded_len and
                           original_tokens[i + 1].whitespace_modifier ==
                           WhitespaceModifier.AFTER_SPACE_OR_SOS) else 0.
                      for i in range(curr_embed.shape[0])])
    )


def update_extra_feats_info(old_extra_feat_info, added_start, added_end):
    (has_been_copy_start, has_been_copy_end, has_been_part_of_copy,
        is_word_start, is_word_end) = old_extra_feat_info
    has_been_copy_start[added_start] = 1
    has_been_part_of_copy[added_start:added_end + 1] = 1
    has_been_copy_end[added_end] = 1
    return has_been_copy_start, has_been_copy_end, has_been_part_of_copy, is_word_start, is_word_end


def get_extra_feat_vec(
    curr_embed,
    extra_feat_info
):
    (has_been_copy_start, has_been_copy_end, has_been_part_of_copy,
     is_word_start, is_word_end) = extra_feat_info
    scale = torch.mean(torch.sum(torch.abs(curr_embed), dim=1))
    extra = torch.transpose(torch.stack((
        (has_been_copy_start * 2 - 1) * scale * 0.1,
        (has_been_copy_end * 2 - 1) * scale * .1,
        (has_been_part_of_copy * 2 - 1) * scale * .1,
        (is_word_start * 2 - 1) * scale * .05,
        (is_word_end * 2 - 1) * scale * .05,
    )), 0, 1)
    return extra


class NBObjectChoiceModel:
    """Naive bayes for modeling an object choice"""
    def __init__(self, type_to_choose: AInixType):
        self._type_to_choose = type_to_choose
        self._type_context = self._type_to_choose.type_context
        self._model = sklearn.naive_bayes.GaussianNB()
        self._object_name_to_ind: Dict[str, int] = {}
        self._num_classes_seen = 0
        # This should not be like this. Only store params for instances that actually appear!
        #self._hacky_all_classes_list = \
        #    list(range(len(self._type_context.get_implementations(self._type_to_choose)) + 1))
        self.all_xs = None
        self.all_ys = []

    def _get_class_ind_for_node(self, node: ObjectChoiceNode, add_if_not_present: bool):
        name_str = "~COPY~" if node.copy_was_chosen else node.get_chosen_impl_name()
        if name_str not in self._object_name_to_ind:
            if add_if_not_present:
                self._object_name_to_ind[name_str] = self._num_classes_seen
                self._num_classes_seen += 1
            else:
                return None
        return self._object_name_to_ind[name_str]

    def add_examples(self, features, nodes: List[ObjectChoiceNode]):
        ys = [self._get_class_ind_for_node(node, True) for node in nodes]
        if self.all_xs is None:
            self.all_xs = features.data.numpy()
        else:
            self.all_xs = np.append(self.all_xs, features.data.numpy(), 0)
        self.all_ys.extend(ys)

    def finalize(self):
        # This method shouldn't exist!
        self._model.fit(self.all_xs, self.all_ys)
        self._ind_to_obj_name = [0] * len(self._object_name_to_ind)
        for v, i in self._object_name_to_ind.items():
            self._ind_to_obj_name[i] = v



def get_nb_learner():
    type_name_to_nb_model: Dict[str, NBObjectChoiceModel] = {}

    def update_fn(new_summary, ast: ObjectChoiceNode):
        for pointer in ast.depth_first_iter():
            node = pointer.cur_node
            if isinstance(node, ObjectChoiceNode):
                type_name = node.type_to_choose.name
                if type_name not in type_name_to_nb_model:
                    type_name_to_nb_model[type_name] = NBObjectChoiceModel(node.type_to_choose)
                new_summary_and_depth = torch.cat(
                    (new_summary, torch.tensor([float(pointer.get_depth())])))
                type_name_to_nb_model[type_name].add_examples(
                    new_summary_and_depth.unsqueeze(0), [node])

    def finalize_fn():
        for model in type_name_to_nb_model.values():
            model.finalize()
        return type_name_to_nb_model

    return update_fn, finalize_fn


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
    summaries, example_refs, example_splits = [], [], []
    nb_update_fn, finalize_nb_fn = get_nb_learner()
    with torch.no_grad():
        for xval in tqdm(list(example_store.get_all_examples())):
            # Get the most prefered y text
            all_y_examples = example_store.get_y_values_for_y_set(xval.y_set_id)
            most_preferable_y = all_y_examples[0]
            new_summary, new_example_ref = _preproc_example(
                xval, most_preferable_y, replacers, embedder, parser, unparser,
                nb_update_fn if xval.split == DataSplits.TRAIN else None)
            summaries.append(new_summary)
            example_refs.append(new_example_ref)
            example_splits.append(xval.split)
    return FullRetModel(
        embedder=embedder,
        summaries=torch.stack(summaries),
        dataset_splits=torch.tensor(example_splits),
        example_refs=np.array(example_refs),
        nb_models=finalize_nb_fn()
    )
