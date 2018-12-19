from typing import Optional, Tuple, List
from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_common.parsing.model_specific.tokenizers import StringTokensMetadata, StringTokenizer
from ainix_common.parsing.stringparser import AstUnparser


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


class CopyInjector:
    """Used to add copy tokens to an AST"""
    pass

def make_copy_versions_of_tree(
    ast: ObjectChoiceNode,
    unparser: AstUnparser,
    token_metadata: StringTokensMetadata
) -> ObjectChoiceNode:
    unparse = unparser.to_string(ast)
    for i, pointer in enumerate(ast.depth_first_iter()):
        if i > 10:
            break

