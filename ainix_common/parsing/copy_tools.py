from typing import Optional, Tuple, List


def string_in_tok_list(string: str, tok_list: List[str]) -> Optional[Tuple[int, int]]:
    """Check to see if a string is substring of a list of tokens. This is like the
    'in' operator of a str, except it takes in consideration that substrings can
    only happen on token boundries.
    Args:
         string: the string to look for
         tok_list: The "joinable" tokens (i.e. no <SPACE> or <SOS> stuff).
    Returns:
         A tuple of the start and end index of the first place this substring
         appears. NOTE: unlike a normal python indexing, the end index is INCLUSIVE.
    # TODO (DNGros): return multiple options if multiple occurances of the string
    """
    def is_valid_start(i: int) -> Optional[int]:
        # Checks to see if index i in the tok_list is a valid start of the
        # string span. Returns the end index if so. None if False.
        if tok_list[i] == "":
            return None
        remaining_str = string
        while remaining_str:
            if remaining_str.startswith(tok_list[i]):
                remaining_str = remaining_str[len(tok_list[i]):]
                i += 1
            else:
                return None
        return i - 1

    for potential_start in range(len(tok_list)):
        potential_end = is_valid_start(potential_start)
        if potential_end is not None:
            return potential_start, potential_end
    return None
