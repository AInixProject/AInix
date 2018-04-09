from program_description import Argument, AIProgramDescription
import random

all_descs = []

### LS ###
aArg = Argument("a", "StoreTrue")
lArg = Argument("l", "StoreTrue")
lsDesc = AIProgramDescription(
    name = "ls", arguments = [aArg, lArg]
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
    ("what files are here with file information", "ls -l")
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
    ("print out what directory am in", "pwd")
]

###########
all_data = lsdata + pwddata
random.shuffle(all_data)

if __name__ == "__main__":
    print("Found %d examples and %d descriptions" % (len(all_data), len(all_descs)))
