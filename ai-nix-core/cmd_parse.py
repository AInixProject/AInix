import bashlex
import program_description
import sys, os
import pudb
import torch
from torch.autograd import Variable

class ArgumentNode():
    def __init__(self, arg, present, value):
        self.arg = arg
        self.present = present
        self.value = value

class ProgramNode():
    def __init__(self, program_desc, arguments, use_cuda):
        self.program_desc = program_desc
        self.arguments = arguments
        
        encodeType = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.arg_present_tensor = Variable(encodeType(
                [1.0 if a.present else 0.0 for a in self.arguments]), requires_grad = False)

    def __repr__(self):
        return "<ProgramNode: " + self.program_desc.name + \
            " arguments=" + str(self.arguments) + ">"

class CmdParseError(Exception):
    pass

class CmdParser(): #bashlex.ast.nodevisitor):
    def __init__(self, avail_desc, use_cuda = False):
        self.command_by_name = {desc.name: desc for desc in avail_desc}
        self.use_cuda = use_cuda
        pass
    
    def _arg_in_word(self, arg, word):
        if len(word) < 2:
            return False
        if word[:1] == "--":
            if word[2:] == arg.name:
                return True
        elif word[0] == "-" and (arg.shorthand is not None or len(arg.name) == 1):
            if (arg.shorthand if arg.shorthand is not None else arg.name) in list(word[1:]):
                return True

        return False
        

    def _parse_node(self, n, cur_parse = []):
        k = n.kind
        if k == 'command':
            parts = n.parts
            if(len(parts) == 0):
                raise CmdParseError("Command has no parts", n)
            commandName = parts.pop(0).word
            if commandName not in self.command_by_name:
                raise CmdParseError("Unrecognized command", commandName)
            command = self.command_by_name[commandName]
            # Figure out all of its arguments
            args = []
            for arg in command.arguments:
                for part in parts:
                    word = part.word
                    pres = self._arg_in_word(arg, word)
                    if pres:
                        args.append(ArgumentNode(arg, True, None))
                        break
                else:
                    args.append(ArgumentNode(arg, False, None))
            return cur_parse + [ProgramNode(command, args, self.use_cuda)]

        else:
            raise CmdParseError("Unexpected kind", k)

    def parse(self, cmd):
        lexedProgram = bashlex.parse(cmd)
        print(lexedProgram)
        return self._parse_node(lexedProgram[0])

    #def visitcommand(self, n, parts):
    #    print("Hey look a command", n, parts)

#parser = CmdParser([])
#parser.parse("ls -a")

def get_descs(filenames):
    out = []
    for fn in filenames:
        with open(fn, 'rb') as f:
            out.append(program_description.load(fn))
