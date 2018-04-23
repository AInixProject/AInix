import bashlex
import program_description
import sys, os
import pudb
import torch
from torch.autograd import Variable
import getopt

class ArgumentNode():
    def __init__(self, arg, present, value):
        self.arg = arg
        self.present = present
        self.value = value

        # Filled in by model
        self.parsed_value = None
        

    def __repr__(self):
        out = "<ArgumentNode: "
        out += self.arg.name
        out += " NOT present" if self.present else " IS present"
        if self.present and self.value is not None:
            out += " val=" + self.value
        return out 
    
    def as_shell_string(self):
        if self.present:
            return self.arg.as_shell_string(self.value)
        else:
            return None

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

    def as_shell_string(self):
        out = self.program_desc.name
        for arg in self.arguments:
            if arg.present:
                out += " " + arg.as_shell_string()
        return out


class JoinNode():
    pass

class PipeNode():
    pass

class CompoundCommandNode():
    def __init__(self):
        self.program_list = []

    def append(self, new_node):
        if isinstance(new_node, CompoundCommandNode):
            self.program_list.extend(new_node.program_list)
        else:
            self.program_list.append(new_node)

    def __getitem__(self, key):
        return self.program_list[key]

    def __len__(self):
        return len(self.program_list)


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
    
    def _getopt(self, cmd_desc, cmd_parts):
        """custom modification of getopt to based off a bashlex ast and stuff like cmd substitution"""
        # build the optstring
        optstring = []
        longargs = []
        for arg in cmd_desc.arguments:
            if arg.position is None:
                needValue = type(arg.argtype).__name__ != "StoreTrue"
                if len(arg.name) == 1:
                    optstring.append(arg.name)
                    if needValue:
                        optstring.append(":")
                else:
                    if arg.shorthand:
                        optstring.append(arg.shorthand)
                    longargs.append(arg.name + "=" if needValue else arg.name)
        optstring = ''.join(optstring)

        # figure out what the input args are
        inargs = [p.word for p in cmd_parts]

        # use normal getopt to parse
        optlist, args = getopt.getopt(inargs, optstring, longargs)

        # convert to a dict
        dictoptlist = {arg:val if val != '' else None for arg, val in optlist}
        
        return dictoptlist, args


    def _parse_node(self, n):
        k = n.kind
        if k == 'command':
            parts = n.parts
            if(len(parts) == 0):
                pudb.set_trace()
                raise CmdParseError("Command has no parts", n)
            commandName = parts.pop(0).word
            if commandName not in self.command_by_name:
                raise CmdParseError("Unrecognized command", commandName)
            command = self.command_by_name[commandName]
            # Figure out all of its arguments
            args = []
            optlist, non_opt_args = self._getopt(command, parts)
            for arg in command.arguments:
                if arg.position is None:
                    if ("-" if len(arg.name) == 1 else "--") + arg.name in optlist:
                        args.append(ArgumentNode(
                            arg, True, optlist[("-" if len(arg.name) == 1 else "--") + arg.name]))
                        continue
                    if arg.shorthand and "-" + arg.shorthand in optlist:
                        args.append(ArgumentNode(arg, True, "-" + optlist["-" + arg.shorthand]))
                        continue
                else:
                    if(len(non_opt_args) > 0):
                        args.append(ArgumentNode(arg, True, " ".join(non_opt_args)))
                        continue
                args.append(ArgumentNode(arg, False, None))
            #cur_parse.append(ProgramNode(command, args, self.use_cuda))
            return ProgramNode(command, args, self.use_cuda)
        elif k == 'pipeline':
            comp = CompoundCommandNode()
            for part in n.parts:
                comp.append(self._parse_node(part))
            return comp
        elif k == 'pipe':
            return PipeNode()
        else:
            raise CmdParseError("Unexpected kind", k)

    def parse(self, cmd):
        lexedProgram = bashlex.parse(cmd)
        n = CompoundCommandNode()
        n.append(self._parse_node(lexedProgram[0]))
        return n

    #def visitcommand(self, n, parts):
    #    print("Hey look a command", n, parts)

#parser = CmdParser([])
#parser.parse("ls -a")

def get_descs(filenames):
    out = []
    for fn in filenames:
        with open(fn, 'rb') as f:
            out.append(program_description.load(fn))
