import pytest
import train
import program_description

lsDesc = program_description.AIProgramDescription(
    name = "ls"
)
pwdDesc = program_description.AIProgramDescription(
    name = "pwd"
)

def test_train():
    data = [
        ("list all files", "ls"),
        ("list all files here", "ls"),
        ("what file am I in", "pwd")
    ]
    train.train(data, [lsDesc, pwdDesc])
