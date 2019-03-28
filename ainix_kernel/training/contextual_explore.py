import os
import torch

from ainix_kernel.models.EncoderDecoder.encdecmodel import make_default_query_encoder, \
    get_default_tokenizers

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    pretrained_checkpoint_path = f"{dir_path}/../../checkpoints/" \
        "lmchkp_30epoch2rnn_merge_toks_total_2.922_ns0.424_lm2.4973.pt"
    (x_tokenizer, query_vocab), y_tokenizer = get_default_tokenizers()
    output_size = 200
    embedder = make_default_query_encoder(x_tokenizer, query_vocab,
                                          output_size, pretrained_checkpoint_path)
    embedder.eval()
    summary1, memory1, toks1 = embedder(
        [
        #"touch source/kernel.bin but only change modified time",
        #"touch bin but only change modified time",
        #"touch check-refs.nix but only change modified time"
        #"touch but only change modified time"]
        #"touch nix but only change modified time"]
        "create a file",
        "make a file"
        #"extract add.tar",
        #"extract difference.tar"
        ]
    )
    print(memory1.shape)
    for t in toks1:
        print([v.token_string for v in t])
    word_i = 0
    #print(torch.cosine_similarity(memory1[0][word_i], memory1[1][word_i], dim=0))
    #print(memory1[0][word_i] - memory1[1][word_i])
    print(torch.cosine_similarity(memory1[0], memory1[1], dim=1))
    print(torch.cosine_similarity(summary1[0], summary1[1], dim=0))

