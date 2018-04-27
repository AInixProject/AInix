from program_description import Argument, AIProgramDescription
import program_description
from custom_fields import Replacer, ReplacementGroup, Replacement
import random
import glob

all_descs = []
for progfile in glob.glob("../ai-nix-programs/*.progdesc.*"):
    with open(progfile, 'r') as f:
        filetext = f.read()
    all_descs.append(program_description.load(filetext))

###########
all_data = []
for desc in all_descs:
    all_data.extend(desc.get_example_tuples()) 
#all_data = lsdata + pwddata + cddata + echodata + rmdata + mkdirData + touchData + catData
random.shuffle(all_data)

filename_repl = ReplacementGroup('FILENAME', Replacement.from_tsv("./data/FILENAME.tsv"))
replacer = Replacer([filename_repl])

def get_all_data_replaced(num_duplicates = 1, num_val_duplicates = None, trainsplit = .7):
    train = all_data[:int(len(all_data)*trainsplit)]*num_duplicates
    if num_val_duplicates is None:
        num_val_duplicates = num_duplicates
    val = all_data[int(len(all_data)*trainsplit):]*num_val_duplicates
    replaced_train = [replacer.strings_replace(*d) for d in train]
    replaced_val = [replacer.strings_replace(*d) for d in val]
    return replaced_train, replaced_val

if __name__ == "__main__":
    print("Found %d examples and %d descriptions" % (len(all_data), len(all_descs)))
