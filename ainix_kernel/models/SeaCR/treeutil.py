from typing import List, Tuple

from ainix_common.parsing.parseast import AstNode, ObjectChoiceNode

def get_type_choice_nodes(
    root_node: AstNode,
    type_name: str
) -> List[Tuple[ObjectChoiceNode, int]]:
    """Gets all the ObjectChoiceNodes that match a certain type inside an AST
    Args:
        root_node: the ast we want to search in
        type_name: the type name choice we are looking for
    Returns:
        List of tuples.
        (A ObjectChoiceNode that is a child of root_node with desired type,
         An int for its depth)
    """
    if not isinstance(type_name, str):
        raise ValueError("Expected string for type_name")
    out = []

    def check_type(cur_node: AstNode, depth: int):
        if cur_node is None:
            return
        if isinstance(cur_node, ObjectChoiceNode):
            if cur_node.get_type_to_choose_name() == type_name:
                out.append((cur_node, depth))
        for child in cur_node.get_children():
            check_type(child, depth + 1)
    check_type(root_node, 0)
    return out