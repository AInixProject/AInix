from ainix_kernel.program_description import Argument, AIProgramDescription
from ainix_kernel import program_description
from ainix_kernel.custom_fields import Replacer, ReplacementGroup, Replacement
from ainix_kernel import tokenizers
import random
import glob

all_descs = []
for progfile in glob.glob("../ai-nix-programs/*.progdesc.*"):
    with open(progfile, 'r') as f:
        filetext = f.read()
    all_descs.append(program_description.load(filetext))

###########

def get_all_data_replaced(num_duplicates = 1, num_val_duplicates = None, trainsplit = .7):
    all_data = []
    for desc in all_descs:
        all_data.extend(desc.get_example_tuples()) 
    random.shuffle(all_data)

    filename_repl = ReplacementGroup('FILENAME', Replacement.from_tsv("./data/FILENAME.tsv"))
    dirname_repl = ReplacementGroup('DIRNAME', Replacement.from_tsv("./data/DIRNAME.tsv"))
    replacer = Replacer([filename_repl, dirname_repl])


    train = all_data[:int(len(all_data)*trainsplit)]*num_duplicates
    if num_val_duplicates is None:
        num_val_duplicates = num_duplicates
    val = all_data[int(len(all_data)*trainsplit):]*num_val_duplicates
    replaced_train = [replacer.strings_replace(*d) for d in train]
    replaced_val = [replacer.strings_replace(*d) for d in val]
    return replaced_train, replaced_val

def get_all_data_from_files(train_nl, train_cmd, val_nl, val_cmd):
    with open(train_nl) as file:
        train_nl = [tokenizers.nonascii_untokenize(l.strip()) for l in file]
    with open(train_cmd) as file:
        train_cmd = [tokenizers.nonascii_untokenize(l.strip()) for l in file]
    with open(val_nl) as file:
        val_nl = [tokenizers.nonascii_untokenize(l.strip()) for l in file]
    with open(val_cmd) as file:
        val_cmd = [tokenizers.nonascii_untokenize(l.strip()) for l in file]
    return zip(train_nl, train_cmd), zip(val_nl, val_cmd)

if __name__ == "__main__":
    print("Found %d examples and %d descriptions" % (len(all_data), len(all_descs)))
    print(list(map(list, get_all_data_from_files("./splits/src-train.txt", "./splits/trg-val.txt",
            "./splits/src-val.txt", "./splits/trg-val.txt"))))
