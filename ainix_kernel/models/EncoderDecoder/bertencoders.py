import functools
from typing import List, Sequence, Tuple

import pytorch_pretrained_bert

from ainix_common.parsing.model_specific import parse_constants, tokenizers
from ainix_common.parsing.model_specific.tokenizers import ModifiedStringToken, \
    StringTokenizerWithMods, StringTokensMetadata, CasingModifier, WhitespaceModifier, \
    get_case_modifier_for_tok, add_pads_to_mod_tokens, add_str_pads, add_pad_arbitrary, MOD_SOS_TOK, \
    MOD_EOS_TOK, merge_tokens, apply_case_mod, MOD_TOK_FOR_MERGE
from ainix_kernel.model_util.vocab import Vocab
from ainix_kernel.models.EncoderDecoder.encoders import QueryEncoder
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

from ainix_kernel.models.LM.cookiemonster import PretrainPoweredQueryEncoder

BERT_SOS_STR = "[CLS]"
BERT_EOS_STR = "[SEP]"
BERT_PAD_STR = "[PAD]"


class ModStringTokenizerFromBert(StringTokenizerWithMods):
    def __init__(self, model_name='bert-base-cased', merge_long_files: bool = True):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.merge_long_files = merge_long_files

    @functools.lru_cache(maxsize=50)
    def tokenize(
        self,
        to_tokenize: str
    ) -> Tuple[List[ModifiedStringToken], StringTokensMetadata]:
        out_toks: List[ModifiedStringToken] = []
        raw_toks = self.bert_tokenizer.tokenize(to_tokenize)
        out_toks.append(MOD_SOS_TOK)
        actual_pos_to_joinable = [None]
        joinable_toks = []
        joinable_toks_to_actual = []

        after_space = True
        str_pointer = 0
        for tok in raw_toks:
            if tok.startswith("##") and not after_space:
                tok = tok[2:]
            while to_tokenize[str_pointer:str_pointer+len(tok)] != tok:
                if to_tokenize[str_pointer] == " ":
                    after_space = True
                    joinable_toks.append(" ")
                    joinable_toks_to_actual.append(None)
                else:
                    raise RuntimeError()
                str_pointer += 1
                if str_pointer + len(tok) > len(to_tokenize):
                    raise RuntimeError()
            casing_mod = get_case_modifier_for_tok(tok, allow_other=True)
            whitespace_mod = WhitespaceModifier.AFTER_SPACE_OR_SOS if after_space \
                else WhitespaceModifier.NOT_AFTER_SPACE
            joinable_toks_to_actual.append(len(out_toks))
            joinable_toks.append(tok)
            tok_str = tok.lower() if casing_mod != CasingModifier.OTHER else tok
            out_toks.append(ModifiedStringToken(tok_str, casing_mod, whitespace_mod))
            actual_pos_to_joinable.append(len(joinable_toks) - 1)
            str_pointer += len(tok)
            after_space = False
        out_toks.append(MOD_EOS_TOK)
        actual_pos_to_joinable.append(None)

        metad = StringTokensMetadata(
            joinable_toks, joinable_toks_to_actual, actual_pos_to_joinable)
        assert len(raw_toks) == len(out_toks[1:-1])

        if self.merge_long_files:
            merge_tokens(out_toks, metad)
        return out_toks, metad


MOD_TO_BERT_MAP = {
    parse_constants.PAD: BERT_PAD_STR,
    parse_constants.EOS: BERT_EOS_STR,
    parse_constants.SOS: BERT_SOS_STR
}
BERT_MERGE_TOK_STR = '##unk'


def mod_toks_to_bert_toks(mod_toks: List[ModifiedStringToken],
                          metad: StringTokensMetadata) -> List[str]:
    outs = []
    for i, mt in enumerate(mod_toks):
        if mt.token_string in MOD_TO_BERT_MAP:
            string = MOD_TO_BERT_MAP[mt.token_string]
        else:
            string = mt.token_string
            string = apply_case_mod(string, mt.casing_modifier)
            is_punc = len(string) == 1 and \
                pytorch_pretrained_bert.tokenization._is_punctuation(string)
            if mt.whitespace_modifier == WhitespaceModifier.NOT_AFTER_SPACE:
                if i <= 1:
                    last_was_punc = False
                else:
                    last_joinable = metad.joinable_tokens[
                        metad.actual_pos_to_joinable_pos[i - 1]]
                    last_was_punc = pytorch_pretrained_bert.tokenization._is_punctuation(
                        last_joinable[-1])


                if not is_punc and not last_was_punc:
                    string = "##" + string
        if mt == MOD_TOK_FOR_MERGE:
            string = BERT_MERGE_TOK_STR
        outs.append(string)
    return outs

class BertEncoder(QueryEncoder):
    def __init__(self):
        super().__init__()
        model_name = 'bert-base-cased'
        self.mod_tokenizer = ModStringTokenizerFromBert(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.eval()

    def forward(
        self,
        queries: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[ModifiedStringToken]]]:
        mod_toks, metads = zip(*self.mod_tokenizer.tokenize_batch(queries))
        raw_toks = [mod_toks_to_bert_toks(mt, meta) for mt, meta in zip(mod_toks, metads)]
        assert len(raw_toks) == len(mod_toks)
        assert len(raw_toks[0]) == len(mod_toks[0])
        raw_toks, lengths = add_str_pads(raw_toks, BERT_PAD_STR)
        mod_toks, mod_tok_lengths = add_pads_to_mod_tokens(mod_toks)
        assert lengths == mod_tok_lengths
        indexed_tokens = [self.mod_tokenizer.bert_tokenizer.convert_tokens_to_ids(t)
                          for t in raw_toks]
        atten_mask = [[0 if tok == BERT_PAD_STR else 1 for tok in toks]
                      for toks in raw_toks]
        tokens_tensor = torch.tensor(indexed_tokens)
        atten_mask_tensor = torch.tensor(atten_mask)
        #tokens_tensor = tokens_tensor.to('cuda')
        #segments_tensors = segments_tensors.to('cuda')
        last_layer, cls_value = self.bert(tokens_tensor,
                                          attention_mask=atten_mask_tensor,
                                          output_all_encoded_layers=False)
        summaries = PretrainPoweredQueryEncoder.sumarize(
            last_layer, lengths, mod_toks, metads)
        embedded_toks = last_layer
        assert len(embedded_toks) == len(queries)
        assert len(embedded_toks[0]) == len(mod_toks[0])

        return torch.Tensor(summaries), torch.Tensor(embedded_toks), mod_toks

    def get_tokenizer(self) -> ModStringTokenizerFromBert:
        return self.mod_tokenizer


if __name__ == "__main__":
    #import torch
    #from pytorch_pretrained_bert import BertTokenizer, BertModel

    ## OPTIONAL: if you want to have more information on what's happening,
    ## activate the logger as follows
    #import logging
    #logging.basicConfig(level=logging.INFO)

    #model_name = 'bert-base-uncased'

    ## Load pre-trained model tokenizer (vocabulary)
    #tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

    ## Tokenized input
    #text = "[CLS] What is lifez?? [SEP]"
    #tokenized_text = tokenizer.tokenize(text)
    #print(tokenized_text)

    ## Convert token to vocabulary indices
    #indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    #segments_ids = [0 for _ in tokenized_text]

    ## Convert inputs to PyTorch tensors
    #tokens_tensor = torch.tensor([indexed_tokens])
    #segments_tensors = torch.tensor([segments_ids])

    ## Load pre-trained model (weights)
    #model = BertModel.from_pretrained(model_name)
    #model.eval()

    ## If you have a GPU, put everything on cuda
    #tokens_tensor = tokens_tensor.to('cuda')
    #segments_tensors = segments_tensors.to('cuda')
    #model.to('cuda')

    ## Predict hidden states features for each layer
    #with torch.no_grad():
    #    encoded_layers, pooled_output = model(tokens_tensor, segments_tensors)
    ## We have a hidden states for each of the 12 layers in model bert-base-uncased
    #assert len(encoded_layers) == 12

    enc = BertEncoder()
    sumarry, emb, toks = enc(["hello there"])
    print(emb)
