from ainix_kernel.models.EncoderDecoder.bertencoders import *


def test_bert_tokenizer():
    tokenizer = ModStringTokenizerFromBert()
    string = "Hello there"
    toks, metad = tokenizer.tokenize(string)
    assert toks == [
        BERT_SOS,
        ModifiedStringToken("hello", CasingModifier.FIRST_UPPER,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedStringToken("there", CasingModifier.LOWER, WhitespaceModifier.AFTER_SPACE_OR_SOS),
        BERT_EOS
    ]
    assert "".join(metad.joinable_tokens) == string
    assert metad.actual_pos_to_joinable_pos == [None, 1, 3, None]
    assert metad.joinable_tokens_pos_to_actual == [1, None, 2]


def test_bert_tokenizer2():
    tokenizer = ModStringTokenizerFromBert()
    string = "helloZ"
    toks, metad = tokenizer.tokenize(string)
    assert toks == [
        BERT_SOS,
        ModifiedStringToken("hello", CasingModifier.LOWER,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedStringToken("z", CasingModifier.SINGLE_CHAR_UPPER,
                            WhitespaceModifier.NOT_AFTER_SPACE),
        BERT_EOS
    ]
    assert "".join(metad.joinable_tokens) == string
    assert metad.actual_pos_to_joinable_pos == [None, 1, 2, None]
    assert metad.joinable_tokens_pos_to_actual == [1, 2]
