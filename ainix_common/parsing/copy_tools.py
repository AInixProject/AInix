from typing import Optional, Tuple, List
from ainix_common.parsing.ast_components import ObjectChoiceNode, CopyNode, \
    ObjectNode, AstObjectChoiceSet, ObjectNodeSet, ImplementationSetData, \
    depth_first_iterate_ast_set_along_path, is_obj_choice_a_not_present_node, AstIterPointer
from ainix_common.parsing.model_specific.tokenizers import StringTokensMetadata, StringTokenizer
from ainix_common.parsing.stringparser import AstUnparser, UnparseResult


def string_in_tok_list(string: str, metadata: StringTokensMetadata) -> Optional[Tuple[int, int]]:
    """Check to see if a string is substring of a list of tokens. This is like the
    'in' operator of a str, except it takes in consideration that substrings can
    only happen on token boundaries.
    Args:
         string: the string to look for
         metadata: the metadata that came from string parser. Used to get the
            joinable_tokens.
    Returns:
         A tuple of the start and end index of the first place this substring
         appears in the original tokenization (not the joinable one).
         NOTE: unlike a normal python indexing, the end index is INCLUSIVE.
    # TODO (DNGros): make version that returns multiple options if multiple
    # ocurances of the string
    """
    # Maybe should left trim the tokens?
    joinable_toks = metadata.joinable_tokens
    mapping = metadata.joinable_tokens_pos_to_actual

    def is_valid_start(i: int) -> Optional[int]:
        # Checks to see if index i in the tok_list is a valid start of the
        # string span. Returns the end index if so. None if False.
        if joinable_toks[i] == "" or mapping[i] is None:
            return None
        remaining_str = string
        while remaining_str:
            if i < len(joinable_toks) and remaining_str.startswith(joinable_toks[i]):
                remaining_str = remaining_str[len(joinable_toks[i]):]
                i += 1
            else:
                return None
        return i - 1

    for potential_start in range(len(joinable_toks)):
        potential_end = is_valid_start(potential_start)
        if potential_end is not None:
            return mapping[potential_start], mapping[potential_end]
    return None


def add_copies_to_ast_set(
    ast: ObjectChoiceNode,
    ast_set: AstObjectChoiceSet,
    unparser: AstUnparser,
    token_metadata: StringTokensMetadata,
    copy_node_weight: float = 1
) -> None:
    """Takes in an AST that has been parsed and adds copynodes where appropriate
    to an AstSet that contains that AST"""
    unparse = unparser.to_string(ast)
    df_ast_nodes = [pointer.cur_node for pointer in ast.depth_first_iter()]
    df_ast_set = list(depth_first_iterate_ast_set_along_path(ast_set, df_ast_nodes))
    assert len(df_ast_nodes) == len(df_ast_set)
    for node, cur_set in zip(df_ast_nodes, df_ast_set):
        if isinstance(node, ObjectChoiceNode):
            # TODO (DNGros): Figure out if we are handling weight and probability right
            # I think works fine now if known valid
            _try_add_copy_node_at_object_choice(
                node, cur_set, True, copy_node_weight, 1, unparse, token_metadata)
        elif isinstance(node, ObjectNode):
            pass
        else:
            raise ValueError("Unrecognized node?")


def _try_add_copy_node_at_object_choice(
    node: ObjectChoiceNode,
    ast_set: AstObjectChoiceSet,
    known_valid: bool,
    max_weight: float,
    max_probability: float,
    unparse: UnparseResult,
    token_metadata: StringTokensMetadata,
):
    if is_obj_choice_a_not_present_node(node):
        return
    if node not in unparse.node_to_span:
        return  # This might be a terrible idea since won't know when bad parser...
    this_node_str = unparse.node_to_string(node)
    copy_pos = string_in_tok_list(this_node_str, token_metadata)
    if copy_pos:
        copy_node = CopyNode(node.type_to_choose, copy_pos[0], copy_pos[1])
        ast_set.add_node_when_copy(copy_node, known_valid, max_weight, max_probability)


def make_copy_versions_of_tree(
    ast: ObjectChoiceNode,
    unparser: AstUnparser,
    token_metadata: StringTokensMetadata
) -> ObjectChoiceNode:
    unparse = unparser.to_string(ast)
    cur_pointer = AstIterPointer(ast, None, None)
    last_pointer = None
    while cur_pointer:
        if isinstance(cur_pointer.cur_node, ObjectChoiceNode):
            this_node_str = unparse.node_to_string(cur_pointer.cur_node)
            copy_pos = string_in_tok_list(this_node_str, token_metadata) if this_node_str else None
            if copy_pos:
                copy_node = CopyNode(
                    cur_pointer.cur_node.type_to_choose, copy_pos[0], copy_pos[1])
                cur_pointer = cur_pointer.dfs_get_next().change_here(copy_node, always_clone=True)
        last_pointer = cur_pointer
        cur_pointer = cur_pointer.dfs_get_next()
    return last_pointer.get_root().cur_node

