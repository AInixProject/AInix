"""Contains tools to look at a utterance, and decide how to execute it.
By how to execute it we mean whether to just run it like a normal
shell command or to run it through our model and run that result."""
import attr
from xonsh.xoreutils import _which
import builtins
from parser import ParseResult

builtin_cmds = ("cd", "echo", "source", "source-bash")

class ExecutionClassifier():
    def classify_string(self, parse : ParseResult): 
        return ExecutionType(
            run_through_model=self.get_whether_to_run_through_model(parse))

    def get_whether_to_run_through_model(self, parse: ParseResult) -> bool:
        if parse.has_force_modeling_escape:
            return True
        firstWordValue = parse.get_first_word()
        if firstWordValue in builtin_cmds:
            return False
        try:
            pathOfProgram = _which.which(firstWordValue, path=builtins.__xonsh__.env['PATH'])
            programOnPath = True
        except _which.WhichError:
            programOnPath = False
        return not programOnPath


@attr.s(frozen = True)
class ExecutionType():
    run_through_model = attr.ib(default=False)
