from program_description import Argument, AIProgramDescription
from custom_fields import Replacer, ReplacementGroup, Replacement
import random

all_descs = []

### LS ###
ls_aArg = Argument("a", "StoreTrue")
ls_AArg = Argument("A", "StoreTrue")
ls_lArg = Argument("l", "StoreTrue")
ls_SArg = Argument("S", "StoreTrue")
ls_sArg = Argument("s", "StoreTrue")
ls_tArg = Argument("t", "StoreTrue")
ls_rArg = Argument("r", "StoreTrue")
ls_iArg = Argument("i", "StoreTrue")
ls_RArg = Argument("R", "StoreTrue")
ls_hArg = Argument("h", "StoreTrue")
ls_XArg = Argument("X", "StoreTrue")
ls_dArg = Argument("X", "StoreTrue")
ls_1Arg = Argument("1", "StoreTrue")
ls_dArg = Argument("d", "StoreTrue")
ls_fileList = Argument("fileList", "Stringlike", position = 1)
lsDesc = AIProgramDescription(
    name = "ls", arguments = [ls_aArg, ls_AArg, ls_lArg, ls_SArg, ls_sArg, ls_tArg, ls_rArg, ls_iArg, ls_RArg, ls_hArg, ls_XArg, ls_dArg, ls_1Arg, ls_dArg, ls_fileList]
)
all_descs.append(lsDesc)
lsdata = [
    ("list all files", "ls"),
    ("print all files and directories here", "ls"),
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
    ("ls with long format", "ls -l"),
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
    ("ls in order of decreasing size", "ls -lS"),
    ("ls order with decreasing file size", "ls -lS"),
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
    ("find inode numbers of files", "ls -i"),
    ("print inode numbers of files here", "ls -i"),
    ("ls with inode numbers", "ls -i"),
    ("list files in home dir", "ls ~"),
    ("what files are here with file information", "ls -l"),
    ("ls recursively down all dirs", "ls -R"),
    ("list files recursively exploring directories", "ls -R"),
    ("ls with latest modified file or directory date as last", "ls -ltr"),
    ("display files in order of size", "ls -lS"),
    ("display files biggest files first", "ls -lS"),
    ("display files smallest files first", "ls -lSr"),
    ("list all files including hidden file starting with '.'", "ls -a"),
    ("list file's inode index number", "ls -i"),
    ("ls -a with inode index number", "ls -ai"),
    ("list with long format - show permissions", "ls -l"),
    ("list long format including hidden files", "ls -la"),
    ("list long format with readable file size", "ls -lh"),
    ("list with long format with file size", "ls -ls"),
    ("list in reverse order", "ls -r"),
    ("ls in reverse order", "ls -r"),
    ("list recursively directory tree", "ls -R"),
    ("list file size", "ls -s"),
    ("ls sort by time & date", "ls -t"),
    ("ls sort by extension name", "ls -X"),
    ("ls group extension together", "ls -X"),
    ("list files grouped by file extension", "ls -X"),
    ("list files sort by extension with hidden files", "ls -Xa"),
    ("list files sort by date/time", "ls -t"),
    ("list all subdirectories", "ls *"),
    ("list only text files", "ls *.txt"),
    ("ls -l sort by file extension", "ls -lX"),
    ("display a list of files and or directories only", "ls"),
    ("display a long list of the content of current directory", "ls -l"),
    ("ls -l with human readable file sizes", "ls -lh"),
    ("list files sort by the largest file size first", "ls -lS"),
    ("ls sort list by extension", "ls -X"),
    ("ls -l sort the list by modification time which the newest first", "ls -lt"),
    ("ls one file per line", "ls -1"),
    ("ls single entry per line", "ls -1"),
    ("show long listing information about each file/directory here", "ls -l"),
    ("ls order files based on last modified time", "ls -t"),
    ("ls -a but dont include . or ..", "ls -A"),
    ("list hidden files but dont include . or ..", "ls -A"),
    ("show all files recursively", "ls -R"),
    ("display files one file per line", "ls -1"),
    ("display total information about Files/Directories", "ls -l"),
    ("display files with file size in human readable form", "ls -lh"),
    ("ls order files based on last modified time in dec-ending order", "ls -lrt"),
    ("list all subdirectories", "ls *"),
    ("ls -r with i-node numbers", "ls -ir"),
    ("list display file Inode number one per line", "ls -i -1"),
    ("ls sort files with size", "ls -lS"),
    ("check inode number of files and directories", "ls -i"),
    # with values
    ("ls all files in pictures", "ls pictures"),
    ("ls all files in pictures and home", "ls pictures ~"),
    ("list all files in ..", "ls .."),
    ("list all jpg files here", "ls *.jpg"),
    ("list all text files here", "ls *.txt"),
    ("list inode numbers of *.h", "ls -i *.h"),
    ("get the inode numbers of *.java", "ls -i *.java"),
    ("list root directory", "ls /"),
    ("list parent directory", "ls .."),
    ("list directories only", "ls -d */"),
    ("list directory entries only", "ls -d */"),
    ("list my home directory", "ls ~"),
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
    ("get my current directory", "pwd"),
    ("print working dir", "pwd"),
    ("get current path", "pwd"),
    ("what directory am I in", "pwd"),
    ("prnt working dir", "pwd"),
    ("print out what directory am in", "pwd"),
    ("get my current directory", "pwd"),
    ("print current working directory", "pwd"),
    ("get my cur path", "pwd"),
]

### CD ###
cd_fileList = Argument("fileList", "Stringlike", position = 1)
cdDesc = AIProgramDescription(
    name = "cd", arguments = [cd_fileList]
)
all_descs.append(cdDesc)
cddata = [
    ("go up one directory", "cd .."),
    ("cd up one directory", "cd .."),
    ("cd to parent dir", "cd .."),
    ("go home", "cd ~"),
    ("cd to home directory", "cd ~"),
    ("go into pictures dir", "cd pictures"),
    ("go up two directories", "cd ../.."),
    ("cd to my home dir", "cd ~"),
    ("change directory to home directory", "cd ~"),
    ("cd to my home directory", "cd ~"),
    ("cd to where I was before", "cd -"),
    ("go back to where I was", "cd -"),
    ("go back to directory was in before last cd", "cd -"),
    ("cd up three directories", "cd ../../.."),
    ("go up to parent directory", "cd .."),
    ("go to tmp directory", "cd /tmp"),
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
    ("print an empty line", "echo"),
    ("output empty line", "echo"),
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
    ("delete file named [-[FILENAME]-]", "rm [-[FILENAME]-]"),
    ("delete [-[FILENAME]-] with interactive deletion", "rm -i [-[FILENAME]-]"),
    ("delete all jpg files here", "rm *.jpg"),
    ("delete all jpg and png files here", "rm *.jpg *.png"),
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
    ("rm *.txt with confirmation", "rm -i *.txt"),
    ("rm all txt files with confirmation", "rm -i *.txt"),
    ("rm *.png with confirmation", "rm -i *.png"),
    ("rm *.jpg *.png *.gif with confirmation", "rm -i *.jpg *.png *.gif"),
    ("rm -rf * with confirmation", "rm -irf *"),
    ("rm -r * with confirmation", "rm -ir *"),
    ("rm [-[FILENAME]-] even if protected", "rm -f [-[FILENAME]-]"),
    ("force delete [-[FILENAME]-]", "rm -f [-[FILENAME]-]"),
    ("rm [-[FILENAME]-] override write protected", "rm -f [-[FILENAME]-]"),
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
    ("make a dir named testdir", "mkdir testdir"),
    ("make a hidden directory named cache", "mkdir .cache"),
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
    ("make a .gitignore file", "touch .gitignore"),
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
    ("print what's in [-[FILENAME]-]", "cat [-[FILENAME]-]"),
    ("print out [-[FILENAME]-]", "cat [-[FILENAME]-]"),
    ("print the contents of [-[FILENAME]-] with line numbers", "cat -n [-[FILENAME]-]"),
    ("write out the contents of [-[FILENAME]-] with line numbers", "cat -n [-[FILENAME]-]"),
    ("print the content of [-[FILENAME]-]", "cat [-[FILENAME]-]"),
    ("output the stuff in [-[FILENAME]-]", "cat [-[FILENAME]-]"),
    ("output the content of [-[FILENAME]-]", "cat [-[FILENAME]-]"),
    ("cat [-[FILENAME]-] with line numbers", "cat -n [-[FILENAME]-]"),
    ("cat the contents of [-[FILENAME]-] with line numbers", "cat -n [-[FILENAME]-]"),
    ("concatenate * together", "cat *"),
    ("concatenate all csv files here together", "cat *.csv"),
    ("concatenate [-[1.FILENAME]-] and [-[2.FILENAME]-] together", "cat [-[1.FILENAME]-] [-[2.FILENAME]-]"),
    ("concatenate [-[1.FILENAME]-], [-[2.FILENAME]-], and [-[3.FILENAME]-] together", "cat [-[1.FILENAME]-] [-[2.FILENAME]-] [-[3.FILENAME]-]"),
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
