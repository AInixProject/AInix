class Argument():
    def __init__(self, arg_name, short_hand):
        self.arg_name = arg_name
        self.short_hand = short_hand

class AIProgramDefinition():
    _arguments = []
    def add_argument(self, argument):
        _arguments.append(argument)
         
    def parse_arguments(self, argument):
        """Parses the provided arguments and returns an actual AI program"""
        return 

def AIProgram():
    class ProgramNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __repr__(self):
            keys = sorted(self.__dict__)
            items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
            return "{}({})".format(type(self).__name__, ", ".join(items))

        def __eq__(self, other):
            return self.__dict__ == other.__dict__
    args = ProgramNamespace()
