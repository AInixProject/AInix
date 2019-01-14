from typing import Sequence, Tuple

import colorama


def get_highlighted_text(
    string: str,
    include_intervals: Sequence[Tuple[int, int]],
    included_exscape = colorama.Fore.BLUE,
    excluded_exscape = colorama.Fore.RESET,
    ending_escape = colorama.Fore.RESET
) -> str:
    """Applies ANSI escape values to highlight text inside certain intervals.
    Args:
        string: The string we wish created a highlighted version of
        include_intervals: A list of (start, end) intervals. Things in the
            interval will be highlighted. For example [(1, 4), (7, 10)] would
            highlight characters [1, 4) and [7, 10)
        included_exscape: The chars to add when starting somethign that is
            in the intervals
        excluded_exscape: The chars to add when starting something outside
            the intervals
        ending_escape: The escape to apply at the end of the string so that way
            the style of the last character won't bleed through
    """
    cur_ind = 0
    out_str_builder = []
    for i, (start, end) in enumerate(include_intervals):
        if start != 0:
            out_str_builder.append(excluded_exscape)
        out_str_builder.append(string[cur_ind:start])
        out_str_builder.append(included_exscape)
        out_str_builder.append(string[start:end])
        cur_ind = end
    if cur_ind != len(string):
        out_str_builder.append(excluded_exscape)
        out_str_builder.append(string[cur_ind:len(string)])
    out_str_builder.append(ending_escape)
    return "".join(out_str_builder)


def get_only_text_in_intervals(
    string: str,
    include_intervals: Sequence[Tuple[int, int]],
    exclude_filler = " "
) -> str:
    """Gets only the parts of string in intervals adding in filler for parts
    not in the intervals
    Args:
        string: The string we wish pull out the parts in the interval
        include_intervals: A list of (start, end) intervals.
        exclude_filler: The char to fill in for parts not in the interval
    """
    cur_ind = 0
    out_str_builder = []
    for i, (start, end) in enumerate(include_intervals):
        out_str_builder.append(exclude_filler * (start - cur_ind))
        out_str_builder.append(string[start:end])
        cur_ind = end
    if cur_ind != len(string):
        out_str_builder.append(exclude_filler * (len(string) - cur_ind))
    return "".join(out_str_builder)
