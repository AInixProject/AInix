from ainix_kernel import constants
def nonascii_tokenizer(input_string):
    out = [[]]
    for c in input_string:
        if not (c <= 'z' and c >= 'A'):
            if out[-1]:
                out.append([])

            if c == " ":
                out[-1].append(constants.SPACE)
            else:
                out[-1].append(c)
            out.append([])
        else:
            out[-1].append(c)
    out = ["".join(toklist) for toklist in out if len(toklist) >= 1]
    return out

def nonascii_untokenize(s):
    s = s.replace(" ", "")
    s = s.replace("<SPACE>", " ")
    return s

def split_tokenization(sequence, 
        split_on = (constants.SPACE, constants.SOS, constants.EOS)):
    """This takes a sequence which is a list of tokens and
    return a list of tuples which are the tokens split on each
    occurance of anything in split_on.
    Note: The split_on tokens will appear as single element tuples
    in the output list."""
    out_groups = []
    newGroup = []
    for e in sequence:
        if e in split_on:
            if newGroup:
                out_groups.append(tuple(newGroup))
            out_groups.append((e,))
            newGroup = []
        else:
            newGroup.append(e)
    if newGroup:
        out_groups.append(tuple(newGroup))
    return out_groups

def quick_tokenizer(input_string):
    out = [[]]
    for c in input_string:
        isalpha = (c <= 'z' and c >= 'A')
        isnum = (c >= '0' and c <= '9')
        isexception = c in ('.', '/')
        if not (isalpha or isnum or isexception):
            if out[-1]:
                out.append([])

            if c == " ":
                out[-1].append(constants.SPACE)
            else:
                out[-1].append(c)
            out.append([])
        else:
            out[-1].append(c)
    out = ["".join(toklist) for toklist in out]
    return out

