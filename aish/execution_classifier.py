"""Contains tools to look at a utterance, and decide how to execute it.
By how to execute it we mean whether to just run it like a normal
shell command or to run it through our model and run that result."""
import attr
from xonsh.xoreutils import _which
import builtins
from parser import ParseResult

class ExecutionClassifier():
    def classify_string(self, parse : ParseResult): 
        firstWordValue = parse.get_first_word()
        try:
            pathOfProgram = _which.which(firstWordValue, path=builtins.__xonsh__.env['PATH'])
            programOnPath = True
        except _which.WhichError:
            programOnPath = False
        return ExecutionType(
            run_through_model=parse.has_force_modeling_escape or not programOnPath)


@attr.s(frozen = True)
class ExecutionType():
    run_through_model = attr.ib(default=False)
