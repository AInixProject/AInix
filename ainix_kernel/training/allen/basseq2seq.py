import itertools

import torch
import torch.optim as optim
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.data.dataset_readers import CopyNetDatasetReader
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.encoder_decoders import CopyNetSeq2Seq
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.nn.activations import Activation
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention, \
    LegacyAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor, Seq2SeqPredictor
from allennlp.training.trainer import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ainix_kernel.training.allen.mycopynet import MyCopyNet
import os

SRC_EMBEDDING_DIM = 300
TGT_EMBEDDING_DIM = 300
HIDDEN_DIM = 128
MAX_DECODING_STEPS = 50
CUDA_DEVICE = 0 if torch.cuda.is_available() else -1

USE_COPY = True

def main():
    target_namespace = "target_tokens"
    if not USE_COPY:
        reader = Seq2SeqDatasetReader(
            source_tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()),
            target_tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()),
            source_token_indexers={'tokens': SingleIdTokenIndexer()},
            target_token_indexers={'tokens': SingleIdTokenIndexer(namespace=target_namespace)}
        )
    else:
        reader = CopyNetDatasetReader(
            source_tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()),
            target_tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()),
            target_namespace=target_namespace
        )
    train_dataset = reader.read('./data/data_train.tsv')
    validation_dataset = reader.read('./data/data_val.tsv')

    vocab = Vocabulary.from_instances(train_dataset,
                                      min_count={'tokens': 3, 'target_tokens': 3})

    en_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=SRC_EMBEDDING_DIM,
        pretrained_file="../opennmt/glove_dir/glove.840B.300d.txt"
    )
    assert en_embedding.weight.requires_grad
    datas = _read_pretrained_embeddings_file(
        en_embedding._pretrained_file, SRC_EMBEDDING_DIM, vocab)
    datas.requires_grad = True
    en_embedding.weight.data = datas
    print(en_embedding.weight.data)
    assert en_embedding.weight.requires_grad
    encoder = PytorchSeq2SeqWrapper(
         torch.nn.LSTM(
             SRC_EMBEDDING_DIM,
             HIDDEN_DIM,
             batch_first=True,
             bidirectional=True,
             dropout=0.3,
             num_layers=1
         )
    )
    #encoder = StackedSelfAttentionEncoder(input_dim=SRC_EMBEDDING_DIM,
    #                                      hidden_dim=HIDDEN_DIM,
    #                                      projection_dim=128, feedforward_hidden_dim=128,
    #                                      num_layers=1, num_attention_heads=8)

    source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})
    attention = DotProductAttention()

    if not USE_COPY:
        model = SimpleSeq2Seq(vocab, source_embedder, encoder, MAX_DECODING_STEPS,
                              target_embedding_dim=TGT_EMBEDDING_DIM,
                              target_namespace='target_tokens',
                              attention=attention,
                              beam_size=8,
                              use_bleu=True)
    else:
        model = MyCopyNet(
            vocab, source_embedder, encoder,
            max_decoding_steps=MAX_DECODING_STEPS,
            target_embedding_dim=TGT_EMBEDDING_DIM,
            target_namespace=target_namespace,
            attention=attention,
            beam_size=8,
            tgt_embedder_pretrain_file="../opennmt/glove_dir/glove.840B.300d.txt"
        )
    model.to(torch.device('cuda'))
    optimizer = optim.Adam(model.parameters())
    iterator = BucketIterator(
        batch_size=64, sorting_keys=[("source_tokens", "num_tokens")],
        padding_noise=0.2
    )

    iterator.index_with(vocab)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        num_epochs=22,
        patience=4,
        serialization_dir="./checkpoints",
        cuda_device=CUDA_DEVICE,
        summary_interval=100
    )
    trainer.train()
    print(en_embedding.weight.data)
    predictor = Seq2SeqPredictor(model, reader)

    # Dump all predictions to a file
    # TODO (DNGros): Is there an automatic way in allennlp to do this??
    pred_toks = []
    with open("pred.txt", "w") as outfile:
        for instance in tqdm(validation_dataset):
            pred = predictor.predict_instance(instance)
            toks = pred['predicted_tokens']
            if toks:
                outfile.write(" ".join(toks[0]) + "\n")
            else:
                outfile.write("" + "\n")


if __name__ == '__main__':
    main()