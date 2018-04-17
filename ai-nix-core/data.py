from program_description import Argument, AIProgramDescription
from custom_fields import Replacer, ReplacementGroup, Replacement
import random

all_descs = []

### LS ###
ls_aArg = Argument("a", "StoreTrue")
ls_lArg = Argument("l", "StoreTrue")
ls_SArg = Argument("S", "StoreTrue")
ls_tArg = Argument("t", "StoreTrue")
ls_rArg = Argument("r", "StoreTrue")
ls_iArg = Argument("i", "StoreTrue")
ls_fileList = Argument("fileList", "Stringlike", position = 1)
lsDesc = AIProgramDescription(
    name = "ls", arguments = [ls_aArg, ls_lArg, ls_SArg, ls_tArg, ls_rArg, ls_iArg, ls_fileList]
)
all_descs.append(lsDesc)
lsdata = [
    ("list all files", "ls"),
    ("list all files here", "ls"),
    ("list my files", "ls"),
    ("list what is here", "ls"),
    ("list files and dirs here", "ls"),
    ("list", "ls"),
    ("list all files with dot files", "ls -a"),
    ("list all files with hidden files", "ls -a"),
    ("list all including dot files", "ls -a"),
    ("ls with dot files", "ls -a"),
    ("ls with hidden files", "ls -a"),
    ("list with dot files", "ls -a"),
    ("list all files with stuff like file size", "ls -l"),
    ("list all files in long format", "ls -l"),
    ("ls with date changed and size", "ls -l"),
    ("show info about files here", "ls -l"),
    ("ls in long format with dot files", "ls -l -a"),
    ("what files are here", "ls"),
    ("list all files here please", "ls"),
    ("list all files and directories here", "ls"),
    ("list all files and dirs in this dir", "ls"),
    ("ls in long format", "ls -l"),
    ("ls sorted by file size", "ls -lS"),
    ("ls biggest files first", "ls -lS"),
    ("list files here biggest files first", "ls -lS"),
    ("list files largest files first", "ls -lS"),
    ("list filese here sorted by files size decending", "ls -lS"),
    ("list files here sorted by file size with dot files", "ls -lSa"),
    ("list files here sorted by modified data", "ls -lt"),
    ("ls sorted newest to oldest", "ls -lt"),
    ("ls sorted oldest to newest", "ls -ltr"),
    ("ls sorted smallest to biggest", "ls -lS"),
    ("ls sorted biggest to smallest", "ls -lSr"),
    ("ls in reverse alphabetical order", "ls -r"),
    ("ls -l sorted by size", "ls -lS"),
    ("ls -l sorted by change data", "ls -lt"),
    ("ls -l sorted by modification data", "ls -lt"),
    ("ls -l sorted by mod data", "ls -lt"),
    ("ls -l sort by mod data", "ls -lt"),
    ("list inode numbers of files here", "ls -i"),
    ("ls with inode numbers", "ls -i"),
    ("list files in home dir", "ls ~"),
    ("what files are here with file information", "ls -l"),
    # with values
    ("ls all files in pictures", "ls pictures"),
    ("ls all files in pictures and home", "ls pictures ~"),
    ("list all files in ..", "ls .."),
    ("list all jpg files here", "ls *.jpg"),
    ("list all text files here", "ls *.txt")
]


### PWD ###
pwdDesc = AIProgramDescription(
    name = "pwd"
)
all_descs.append(pwdDesc)

pwddata = [
    ("what file am I in", "pwd"),
    ("print working directory", "pwd"),
    ("where am I", "pwd"),
    ("print current dir", "pwd"),
    ("what file am I in", "pwd"),
    ("print working dir", "pwd"),
    ("get current path", "pwd"),
    ("what directory am I in", "pwd"),
    ("prnt working dir", "pwd"),
    ("print out what directory am in", "pwd"),
    ("get my current directory", "pwd")
]

### CD ###
cd_fileList = Argument("fileList", "Stringlike", position = 1)
cdDesc = AIProgramDescription(
    name = "cd", arguments = [cd_fileList]
)
all_descs.append(cdDesc)
cddata = [
    ("go up one directory", "cd .."),
    ("go home", "cd ~"),
    ("cd to home directory", "cd ~"),
    ("go into pictures dir", "cd pictures"),
    ("go up two directories", "cd ../.."),
    ("cd to my home dir", "cd ~"),
    ("cd to my home directory", "cd ~"),
    ("cd to where I was before", "cd -"),
    ("go back to where I was", "cd -"),
]

### ECHO ###
echo_fileList = Argument("fileList", "Stringlike", position = 1)
echoDesc = AIProgramDescription(
    name = "echo", arguments = [echo_fileList]
)
all_descs.append(echoDesc)
echodata = [
    ("say hello", "echo hello"),
    ("print out \"hello world\"", "echo hello world"),
    ("print out hello world", "echo hello world"),
    ("print an empty new line", "echo"),
    ("print an empty line", "echo")
]

### RM ###
rm_rArg = Argument("r", "StoreTrue")
rm_fArg = Argument("f", "StoreTrue")
rm_iArg = Argument("i", "StoreTrue")
rm_fileList = Argument("fileList", "Stringlike", position = 1)
rmDesc = AIProgramDescription(
    name = "rm", arguments = [rm_rArg, rm_fArg, rm_iArg, rm_fileList]
)
all_descs.append(rmDesc)
rmdata = [
    ("delete everything here", "rm -r *"),
    ("force delete everything here", "rm -rf *"),
    ("remove everything here", "rm -r *"),
    ("delete [-[FILENAME]-]", "rm [-[FILENAME]-]"),
    ("delete all jpg files here", "rm *.jpg"),
    ("delete all png files here", "rm *.png"),
    ("delete all text files here", "rm *.txt"),
    ("remove all txt files", "rm *.txt"),
    ("remove all c files", "rm *.c"),
    ("delete all header files here", "rm *.h"),
    ("delete all md files here", "rm *.md"),
    ("remove all markdown files", "rm *.md"),
    ("rm everything here with confirmation first", "rm -ri *"),
    ("rm everything here but ask me first", "rm -ri *"),
    ("delete all jpg files with confirmation first", "rm -i *.jpg"),
    ("delete [-[1.FILENAME]-] and [-[2.FILENAME]-]", "rm [-[1.FILENAME]-] [-[2.FILENAME]-]"),
    ("remove [-[1.FILENAME]-] and [-[2.FILENAME]-]", "rm [-[1.FILENAME]-] [-[2.FILENAME]-]"),
    ("remove [-[1.FILENAME]-], [-[2.FILENAME]-] and [-[3.FILENAME]-]", "rm [-[1.FILENAME]-] [-[2.FILENAME]-] [-[3.FILENAME]-]"),
    ("rm * with confirmation", "rm -i *"),
    ("rm -rf * with confirmation", "rm -irf *"),
    ("rm -r * with confirmation", "rm -ir *"),
    ("rm everything here with prompt before removal", "rm -ir *"),
]

### mkdir ###
mkdir_fileList = Argument("fileList", "Stringlike", position = 1)
mkdirDesc = AIProgramDescription(
    name = "mkdir", arguments = [mkdir_fileList]
)
all_descs.append(mkdirDesc)
mkdirData = [
    ("make a directory named foo", "mkdir foo"),
    ("make a foo directory", "mkdir foo"),
    ("make a directory named testdir", "mkdir testdir"),
    ("make a dir named testdir", "mkdir testdir")
]

### touch ###
touch_rArg = Argument("r", "StoreTrue")
touch_fileList = Argument("fileList", "Stringlike", position = 1)
touchDesc = AIProgramDescription(
    name = "touch", arguments = [touch_fileList, touch_rArg]
)
all_descs.append(touchDesc)
touchData = [
    ("make a file named [-[FILENAME]-]", "touch [-[FILENAME]-]"),
    ("create a file named [-[FILENAME]-]", "touch [-[FILENAME]-]"),
    ("make a file named __init__.py", "touch __init__.py"),
    ("create a __init__.py", "touch __init__.py"),
    ("make this a python module", "touch __init__.py"),
    ("setup this dir as a python module", "touch __init__.py"),
    ("make a blank gitignore", "touch .gitignore"),
    ("make a gitignore", "touch .gitignore"),
    ("make an empty gitignore", "touch .gitignore"),
    ("create an empty file named [-[FILENAME]-]", "touch [-[FILENAME]-]"),
    ("add empty file named [-[FILENAME]-] here", "touch [-[FILENAME]-]"),
    ("set the last mod time of [-[FILENAME]-] to now", "touch [-[FILENAME]-]"),
    ("set the last accessed and modified time of [-[FILENAME]-] to now", "touch [-[FILENAME]-]"),
    ("reset the last modified time of [-[FILENAME]-] to now", "touch [-[FILENAME]-]"),
    ("set the last modified time of [-[1.FILENAME]-] to be the same as [-[2.FILENAME]-]", "touch -r [-[2.FILENAME]-] [-[1.FILENAME]-]"),
    ("touch [-[1.FILENAME]-] using last modified and access times of [-[2.FILENAME]-]", "touch -r [-[2.FILENAME]-] [-[1.FILENAME]-]"),
]

### cat ###
cat_nArg = Argument("n", "StoreTrue")
cat_fileList = Argument("fileList", "Stringlike", position = 1)
catDesc = AIProgramDescription(
    name = "cat", arguments = [cat_nArg, cat_fileList]
)
all_descs.append(catDesc)
catData = [
    ("print the contents of [-[FILENAME]-]", "cat [-[FILENAME]-]"),
    ("print whats in [-[FILENAME]-]", "cat [-[FILENAME]-]"),
    ("print out [-[FILENAME]-]", "cat [-[FILENAME]-]"),
    ("print the contents of [-[FILENAME]-] with line numbers", "cat -n [-[FILENAME]-]"),
    ("write out the contents of [-[FILENAME]-] with line numbers", "cat -n [-[FILENAME]-]"),
    ("print the content of [-[FILENAME]-]", "cat -n [-[FILENAME]-]"),
    ("concatenate * together", "cat *"),
    ("concatenate [-[1.FILENAME]-] and [-[2.FILENAME]-] together", "cat [-[1.FILENAME]-] [-[2.FILENAME]-]"),
]

###########
all_data = lsdata + pwddata + cddata + echodata + rmdata + mkdirData + touchData + catData
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
