from ainix_kernel.models.EncoderDecoder.bertencoders import *


def test_bert_tokenizer():
    tokenizer = ModStringTokenizerFromBert()
    string = "Hello there"
    toks, metad = tokenizer.tokenize(string)
    assert toks == [
        MOD_SOS_TOK,
        ModifiedStringToken("hello", CasingModifier.FIRST_UPPER,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedStringToken("there", CasingModifier.LOWER, WhitespaceModifier.AFTER_SPACE_OR_SOS),
        MOD_EOS_TOK
    ]
    assert "".join(metad.joinable_tokens) == string
    assert metad.actual_pos_to_joinable_pos == [None, 0, 2, None]
    assert metad.joinable_tokens_pos_to_actual == [1, None, 2]


def test_bert_tokenizer2():
    tokenizer = ModStringTokenizerFromBert()
    string = "helloZ"
    toks, metad = tokenizer.tokenize(string)
    assert toks == [
        MOD_SOS_TOK,
        ModifiedStringToken("hello", CasingModifier.LOWER,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedStringToken("z", CasingModifier.SINGLE_CHAR_UPPER,
                            WhitespaceModifier.NOT_AFTER_SPACE),
        MOD_EOS_TOK
    ]
    assert "".join(metad.joinable_tokens) == string
    assert metad.actual_pos_to_joinable_pos == [None, 0, 1, None]
    assert metad.joinable_tokens_pos_to_actual == [1, 2]


def test_conversion():
    test_str = "hello There testZh YouTube?"
    tokenizer = ModStringTokenizerFromBert()
    assert mod_toks_to_bert_toks(*tokenizer.tokenize(test_str)) == \
        [BERT_SOS_STR] + tokenizer.bert_tokenizer.tokenize(test_str) + [BERT_EOS_STR]


def test_conversion2():
    test_str = "'hello' There's #>?"
    tokenizer = ModStringTokenizerFromBert()
    assert mod_toks_to_bert_toks(*tokenizer.tokenize(test_str)) == \
           [BERT_SOS_STR] + tokenizer.bert_tokenizer.tokenize(test_str) + [BERT_EOS_STR]


def test_conversion3():
    test_str = "hello/foo/bar/baz.text"
    tokenizer = ModStringTokenizerFromBert()
    assert mod_toks_to_bert_toks(*tokenizer.tokenize(test_str)) ==\
           ['[CLS]', 'hello', BERT_MERGE_TOK_STR, 'text', '[SEP]']

def test_conversion4():
    test_str = "hello/foo/bar/baztext"
    tokenizer = ModStringTokenizerFromBert()
    assert mod_toks_to_bert_toks(*tokenizer.tokenize(test_str)) == \
           ['[CLS]', 'hello', BERT_MERGE_TOK_STR, '##text', '[SEP]']
