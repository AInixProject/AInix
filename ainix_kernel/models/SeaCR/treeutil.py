from typing import List, Tuple

from ainix_common.parsing.ast_components import AstNode, ObjectChoiceNode

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

#def _is_like_a_obj_choice_node(node: AstNode):
#    """Do some wonky python import stuff I don't feel like debugging right
#    now isinstance(node, ObjectChoiceNode) isn't working. Instead we are
#    just going full on duck-typing mode and just checking to see if it has
#    a way to access it's type name to choose."""
#    #return hasattr(node, "get_type_to_choose_name")
#    return isinstance(node, ObjectChoiceNode)



