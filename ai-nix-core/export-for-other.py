import data as sampledata
from cmd_parse import CmdParser, CmdParseError
import pudb


SPACE = "<SPACE>"
def nonascii_tokenizer(input_string):
    out = [[]]
    for c in input_string:
        if not (c <= 'z' and c >= 'A'):
            if out[-1]:
                out.append([])

            if c == " ":
                out[-1].append(SPACE)
            else:
                out[-1].append(c)
            out.append([])
        else:
            out[-1].append(c)
    out = ["".join(toklist) for toklist in out]
    return out

parser = CmdParser(sampledata.all_descs)
def as_parallel_strings(data):
    nl, cmd = zip(*data)
    cmdlist = []
    for c in cmd:
        try:
            ast = parser.parse(c)
        except CmdParseError as e:
            print("parse failed on ", c, e)
        as_str = ast[0].as_shell_string()
        tokenized = nonascii_tokenizer(as_str)
        cmdlist.append(" ".join(tokenized))
    nl = [" ".join(nonascii_tokenizer(line)) for line in nl]
    def flatten(l):
        out = []
        for line in l:
            for word in line.split():
                out.append(word)
        return out
    vocab = set(flatten(nl + cmdlist))
    return '\n'.join(nl), '\n'.join(cmdlist), '\n'.join(vocab)

def write_to_file(s, filename):
    with open(filename, "w") as text_file:
        text_file.write(s)

if __name__ == "__main__":
    num_train_duplicates = 5
    print("loading data")
    train, val = sampledata.get_all_data_replaced(num_train_duplicates,2)
    print("processing train")
    src_train, trg_train, train_vocab = as_parallel_strings(train)
    print("writing train")
    write_to_file(src_train, "src-train.txt") 
    write_to_file(trg_train, "trg-train.txt") 
    write_to_file(train_vocab, "train-vocab.txt") 
    print("processing val")
    src_val, trg_val, _ = as_parallel_strings(val)
    print("writing val")
    write_to_file(src_val, "src-val.txt") 
    write_to_file(trg_val, "trg-val.txt") 
    print("done.")
