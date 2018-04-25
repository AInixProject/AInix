SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

COPY_TOKENS = tuple(["COPY_P%d" % c for c in range(10)])
# This is the max number of nodes in a compound command. 
# It includes joinnodes (like pipe), and a EndOfCommandNode.
# ex:  foo | bar -> 4 nodes
MAX_COMMAND_LEN = 8
