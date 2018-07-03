from xonsh import built_ins as xonsh_builtin
from parser import ParseResult
def _convert_to_xonsh_subproc_string(parse : ParseResult):
    out = [[]]
    WORDS_THAT_SHOULD_BE_ALONE = ('|', '>>', '>', '<', '<<', '&&', '||')
    for word in parse.words:
        if word in WORDS_THAT_SHOULD_BE_ALONE:
            out.append(word)
            out.append([])
        else:
            out[-1].append(word)
    return out


def execute(parse : ParseResult):
    xonsh_seq = _convert_to_xonsh_subproc_string(parse)
    xonsh_builtin.run_subproc(xonsh_seq)

