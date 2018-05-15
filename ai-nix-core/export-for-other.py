import data as sampledata
from cmd_parse import CmdParser, CmdParseError
import tokenizers
import pudb

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
        tokenized = tokenizers.nonascii_tokenizer(as_str)
        cmdlist.append(" ".join(tokenized))
    nl = [" ".join(tokenizers.nonascii_tokenizer(line)) for line in nl]
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
    write_to_file(src_train, "splits/src-train.txt") 
    write_to_file(trg_train, "splits/trg-train.txt") 
    write_to_file(train_vocab, "splits/train-vocab.txt") 
    print("processing val")
    src_val, trg_val, _ = as_parallel_strings(val)
    print("writing val")
    write_to_file(src_val, "splits/src-val.txt") 
    write_to_file(trg_val, "splits/trg-val.txt") 
    print("done.")
