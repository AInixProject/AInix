from ainix_kernel.models.SeaCR.comparer import Comparer
from models.SeaCR.comparer import ComparerResult
from parseast import ObjectChoiceNode, AstNode


class TorchComparer(Comparer):
    def __init__(self):
        pass

    def compare(
            self,
            gen_query: str,
            gen_ast_current_leaf: ObjectChoiceNode,
            current_gen_depth: int,
            example_query: str,
            example_ast_root: AstNode,
    ) -> ComparerResult:
        pass

    def train(
            self,
            gen_query: str,
            gen_ast_current_leaf: ObjectChoiceNode,
            current_gen_depth: int,
            example_query: str,
            example_ast_root: AstNode,
            expected_result: ComparerResult
    ):
        pass
