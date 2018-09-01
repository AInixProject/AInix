from abc import ABC, abstractmethod
from models.model_types import StringTypeTranslateCF
from indexing.exampleindex import ExamplesIndex, Example
from ainix_common.parsing.parseast import AstNode, ObjectNode, \
    ObjectChoiceNode, ArgPresentChoiceNode, StringParser
from ainix_common.parsing.typecontext import AInixType, AInixObject
from typing import List


class SeaCRModel(StringTypeTranslateCF):
    def __init__(
        self,
        index: ExamplesIndex,
        type_predictor: 'TypePredictor' = None,
        arg_present_predictor: 'ArgPresentPredictor' = None
    ):
        self.type_context = index.type_context
        self.type_predictor = type_predictor if type_predictor else TypePredictor(index)
        if arg_present_predictor:
            self.arg_present_predictor = arg_present_predictor
        else:
            self.arg_present_predictor = ArgPresentPredictor(index, self.type_predictor)

    def predict(self, x_string: str, y_type_name: str):
        root_type = self.type_context.get_type_by_name(y_type_name)
        root_node = ObjectChoiceNode(root_type, None)
        self.predict_step(x_string, root_node)
        return root_node

    def predict_step(self, x_query: str, current_leaf: AstNode, current_depth: int = 0):
        if isinstance(current_leaf, ObjectChoiceNode):
            next_nodes = [self.type_predictor.predict(x_query, current_leaf)]
        elif isinstance(current_leaf, ObjectNode):
            next_nodes = current_leaf.get_children()
        elif isinstance(current_leaf, ArgPresentChoiceNode):
            next_nodes = [self.arg_present_predictor.predict(
                x_query, current_leaf, current_depth)]
        else:
            raise ValueError(f"leaf node {current_leaf} not predictable")
        next_nodes = [node for node in next_nodes if node is not None]

        for next_node in next_nodes:
            self.predict_step(x_query, next_node, current_depth+1)


class TypePredictor:
    def __init__(self, index: ExamplesIndex):
        self.index = index

    def _get_impl_names_from_example(
        self,
        example: Example,
        type_to_choose: str
    ) -> List[str]:
        # TODO (DNGros): This is disgusting and shouldnt exist
        out = []
        for s in example.yindexable.split(f"CLASSIFY_TYPE={type_to_choose}")[1:]:
            o_rep = s.split()[1]
            o_name = o_rep.split("=")[1]
            out.append(o_name)
        return out

    def eval_single_example(self, x_query, type_name):
        pass

    def predict_on_type_name(self, x_query: str, type_name: str) -> str:
        search_results = self.index.get_nearest_examples(x_query, choose_type_name=type_name)
        potential_types = self._get_impl_names_from_example(search_results[0], type_name)
        choose = potential_types[0]
        return choose

    def predict(self, x_query: str, current_leaf: ObjectChoiceNode) -> ObjectNode:
        type_context = current_leaf.type_to_choose.type_context
        type_name = current_leaf.type_to_choose.name
        choose_name = self.predict_on_type_name(x_query, type_name)
        choose_impl = type_context.get_object_by_name(choose_name)
        return current_leaf.add_valid_choice(choose_impl, 1)


class ArgPresentPredictor:
    MAX_DEPTH = 10

    def __init__(self, index: ExamplesIndex, type_predictor: TypePredictor):
        self.index = index
        self.type_predictor = type_predictor

    def predict(
        self,
        x_query: str,
        current_leaf: ArgPresentChoiceNode,
        current_depth: int
    ) -> ObjectChoiceNode:
        if current_depth > self.MAX_DEPTH:
            # If we reached max depth just try and terminate
            return current_leaf.set_choice(False)
        predicted_impl_name = self.type_predictor.predict_on_type_name(
            x_query=x_query,
            type_name=current_leaf.argument.present_choice_type_name
        )
        # TODO (DNGros): disgusting hardcoded strings
        if predicted_impl_name.endswith("NOTPRESENT"):
            return current_leaf.set_choice(False)
        elif predicted_impl_name.endswith("PRESENT"):
            return current_leaf.set_choice(True)
        else:
            raise RuntimeError("wtf")


class Comparer(ABC):
    pass
