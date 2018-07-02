from xonsh import built_ins as xonsh_builtin
from parser import ParseResult
def _convert_to_xonsh_subproc_string(parse : ParseResult):
    return [parse.words]

def execute(parse : ParseResult):
    xonsh_seq = _convert_to_xonsh_subproc_string(parse)
    xonsh_builtin.run_subproc(xonsh_seq)

