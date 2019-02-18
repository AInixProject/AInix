# SOS and EOS stands for Start of Sentence and End of Sentence.
# However, this is a bit of a misnomer. It is placed at the start and end of an
# input utterace, which might be several sentences long.
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"
SPACE = "<SPACE>"
PAD = "<PAD>"
# Splits two parts of task. In the language model it splits two sentence.
# it is also envisionsed it could be used to split (Question, Passage) in a Q&A
# task or between (Text, Hypothesis) in SNLI or something...
TASK_SPLITTER = "<TASK_SPLITTER>"
# A task token is a generic token to represent something special should be
# done here. In the language modeling task it takes the place of a <MASK> token.
TASK_TOK = "<TASK_TOK>"

ALL_SPECIALS = [SOS, EOS, UNK, SPACE, PAD, TASK_SPLITTER, TASK_TOK]
TOKEN_SPECIALS = [PAD, TASK_SPLITTER, TASK_TOK]
