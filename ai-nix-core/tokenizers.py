import constants
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
    out = ["".join(toklist) for toklist in out]
    return out

def nonascii_untokenize(s):
    s = s.replace(" ", "")
    s = s.replace("<SPACE>", " ")
    return s

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

