"""This module defines the SeaCR (Search Compare Recurse) model"""
from typing import Type

from ainix_kernel.indexing.examplestore import ExamplesStore
from ainix_kernel.models.SeaCR.comparer import SimpleRulebasedComparer, OracleComparer
from ainix_kernel.models.model_types import StringTypeTranslateCF
from ainix_kernel.indexing.exampleindex import ExamplesIndex
from ainix_common.parsing.ast_components import AstNode, ObjectNode, \
    ObjectChoiceNode, AstObjectChoiceSet, ObjectNodeSet
from ainix_common.parsing.stringparser import StringParser
from ainix_common.parsing.typecontext import AInixType, TypeContext
from ainix_kernel.model_util.tokenizers import NonAsciiTokenizer, AstStringTokenizer
from ainix_kernel.model_util.vocab import CounterVocabBuilder, make_vocab_from_example_store

from ainix_kernel.models.SeaCR.type_predictor import SearchingTypePredictor, TypePredictor, \
    TerribleSearchTypePredictor


def make_rulebased_seacr(index: ExamplesIndex):
    """A factory helper method which makes a SeaCR model which is deterministic
    and does not require any training."""
    return SeaCRModel(index, SearchingTypePredictor(index, comparer=SimpleRulebasedComparer()))

def _get_default_tokenizers():
    """Returns tuple (default x tokenizer, default y tokenizer)"""
    return NonAsciiTokenizer(), AstStringTokenizer()


DEFAULT_WORD_VECTOR_SIZE = 64


def _make_default_seacr_with_type_predictor(index: ExamplesIndex,
                                            type_predictor_type: Type[TypePredictor]):
    from ainix_kernel.models.SeaCR import torchcomparer
    x_tokenizer, y_tokenizer = _get_default_tokenizers()
    x_vocab, y_vocab = make_vocab_from_example_store(
        index, x_tokenizer, y_tokenizer)
    comparer = torchcomparer.get_default_torch_comparer(
        x_vocab=x_vocab,
        y_vocab=y_vocab,
        x_tokenizer=x_tokenizer,
        y_tokenizer=y_tokenizer,
        out_dims=DEFAULT_WORD_VECTOR_SIZE
    )
    type_predictor = type_predictor_type(index, comparer=comparer)
    return SeaCRModel(index, type_predictor)


def make_default_seacr(index: ExamplesIndex) -> 'SeaCRModel':
    """A factory helper method which makes the current standard SeaCR model."""
    return _make_default_seacr_with_type_predictor(index, SearchingTypePredictor)


def make_default_seacr_no_search(index: ExamplesIndex) -> 'SeaCRModel':
    """A factory which makes a SeaCR model, but without a searching capability.
    Must instead compare every entry, and rely on the comparer."""
    return _make_default_seacr_with_type_predictor(index, TerribleSearchTypePredictor)


def make_default_seacr_with_oracle_comparer(index: ExamplesIndex):
    """Factory which constructs the current standard SeaCR model but using an
    oracle as the comparer (Useful for testing the other parts of the model
    without depending on the comparer learning stuff)."""
    comparer = OracleComparer(index)
    return SeaCRModel(index, SearchingTypePredictor(index, comparer=comparer))


class SeaCRModel(StringTypeTranslateCF):
    def __init__(
        self,
        index: ExamplesIndex,
        type_predictor: 'TypePredictor',
    ):
        self.type_context = index.type_context
        self.type_predictor = type_predictor

    def predict(
        self,
        x_string: str,
        y_type_name: str,
        use_only_train_data: bool
    ) -> AstNode:  # TODO (DNGros): change to set
        root_type = self.type_context.get_type_by_name(y_type_name)
        root_node = ObjectChoiceNode(root_type)
        self._predict_step(x_string, root_node, root_node, root_type, use_only_train_data, 0)
        root_node.freeze()
        return root_node

    def _predict_step(
        self,
        x_query: str,
        current_root: ObjectChoiceNode,
        current_leaf: AstNode,
        root_y_type: AInixType,
        use_only_train_data: bool,
        current_depth: int
    ):
        if isinstance(current_leaf, ObjectChoiceNode):
            predicted_impl = self.type_predictor.predict(
                x_query, current_root, current_leaf, use_only_train_data,
                current_depth)
            new_node = ObjectNode(predicted_impl)
            current_leaf.set_choice(new_node)
            if new_node is not None:
                self._predict_step(x_query, current_root, new_node, root_y_type,
                                   use_only_train_data, current_depth + 1)
        elif isinstance(current_leaf, ObjectNode):
            # TODO (DNGros): this is messy. Should have better iteration based
            # off next unfilled node rather than having to mutate state.
            #####
            # Actually it would probably be better to just store a pointer to
            # the root and a pointer to the leaf. Then you call an add_to_leaf()
            # on the root which does a copying add, with structure sharing of
            # any arg before the arg that leaf is on. This is also nice because
            # then AstNodes can be made purly immutable and beam search becomes
            # nearly trivial to do
            for arg in current_leaf.implementation.children:
                if not arg.required:
                    new_node = ObjectChoiceNode(arg.present_choice_type)
                elif arg.type is not None:
                    new_node = ObjectChoiceNode(arg.type)
                else:
                    continue
                current_leaf.set_arg_value(arg.name, new_node)
                self._predict_step(x_query, current_root, new_node, root_y_type,
                                   use_only_train_data, current_depth + 1)
        else:
            raise ValueError(f"leaf node {current_leaf} not predictable")

    def _train_obj_node_step(
        self,
        x_query: str,
        expected: ObjectNodeSet,
        current_gen_root: ObjectChoiceNode,
        current_gen_leaf: ObjectNode,
        teacher_force_path: ObjectNode,
        current_depth: int
    ):
        arg_set_data = expected.get_arg_set_data(teacher_force_path.as_childless_node())
        assert arg_set_data is not None, "Teacher force path not in expected ast set!"
        for arg in teacher_force_path.implementation.children:
            #if arg.type is None:
            #    continue
            next_choice_set = arg_set_data.arg_to_choice_set[arg.name]
            # TODO (DNGros): This is currently somewhat gross as it relies on the _train_step
            # call mutating state. Once it is changed to make changes on current_gen_root
            # this shouldn't be an issue.
            next_gen_leaf = ObjectChoiceNode(arg.next_choice_type)
            current_gen_leaf.set_arg_value(arg.name, next_gen_leaf)
            self._train_step(x_query, next_choice_set, current_gen_root, next_gen_leaf,
                             teacher_force_path.get_choice_node_for_arg(arg.name), current_depth+1)

    def _train_step(
        self,
        x_query: str,
        expected: AstObjectChoiceSet,
        current_gen_root: ObjectChoiceNode,
        current_gen_leaf: ObjectChoiceNode,
        teacher_force_path: ObjectChoiceNode,
        current_depth: int
    ):  # TODO This should likely eventually return the new current_gen ast
        self.type_predictor.train(x_query, current_gen_root, current_gen_leaf,
                                  expected, current_depth)
        # figure out where going next
        next_expected_node = expected.get_next_node_for_choice(
            teacher_force_path.get_chosen_impl_name())
        assert next_expected_node is not None, "Teacher force path not in expected ast set!"
        next_object_node = ObjectNode(teacher_force_path.next_node.implementation)
        current_gen_leaf.set_choice(next_object_node)
        self._train_obj_node_step(x_query, next_expected_node, current_gen_root,
                                  next_object_node, teacher_force_path.next_node,
                                  current_depth+1)

    def train(
        self,
        x_string: str,
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode
    ) -> None:
        current_gen_node = ObjectChoiceNode(teacher_force_path.type_to_choose)
        self._train_step(
            x_string, y_ast, current_gen_node, current_gen_node, teacher_force_path, 0
        )

    @classmethod
    def make_examples_store(cls, type_context: TypeContext, is_training) -> ExamplesIndex:
        return ExamplesIndex(type_context,
                             ExamplesIndex.get_default_ram_backend() if is_training else None)

