import functools
from typing import List, Sequence, Tuple

from ainix_common.parsing.model_specific import parse_constants, tokenizers
from ainix_common.parsing.model_specific.tokenizers import ModifiedStringToken, \
    StringTokenizerWithMods, StringTokensMetadata, CasingModifier, WhitespaceModifier, \
    get_case_modifier_for_tok, add_pads_to_mod_tokens
from ainix_kernel.model_util.vocab import Vocab
from ainix_kernel.models.EncoderDecoder.encoders import QueryEncoder
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

BERT_SOS = ModifiedStringToken("[CLS]", CasingModifier.CASELESS,
                               WhitespaceModifier.AFTER_SPACE_OR_SOS)
BERT_EOS = ModifiedStringToken("[SEP]", CasingModifier.CASELESS,
                               WhitespaceModifier.AFTER_SPACE_OR_SOS)


class ModStringTokenizerFromBert(StringTokenizerWithMods):
    def __init__(self, model_name='bert-base-cased'):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

    @functools.lru_cache(maxsize=50)
    def tokenize(
        self,
        to_tokenize: str
    ) -> Tuple[List[ModifiedStringToken], StringTokensMetadata]:
        out_toks: List[ModifiedStringToken] = []
        raw_toks = self.bert_tokenizer.tokenize(to_tokenize)
        out_toks.append(BERT_SOS)
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
            casing_mod = get_case_modifier_for_tok(tok)
            whitespace_mod = WhitespaceModifier.AFTER_SPACE_OR_SOS if after_space \
                else WhitespaceModifier.NOT_AFTER_SPACE
            joinable_toks_to_actual.append(len(out_toks))
            joinable_toks.append(tok)
            out_toks.append(ModifiedStringToken(tok.lower(), casing_mod, whitespace_mod))
            actual_pos_to_joinable.append(len(joinable_toks))
            str_pointer += len(tok)
            after_space = False
        out_toks.append(BERT_EOS)
        actual_pos_to_joinable.append(None)

        metad = StringTokensMetadata(
            joinable_toks, joinable_toks_to_actual, actual_pos_to_joinable)

        assert len(raw_toks) == len(out_toks[1:-1])
        return out_toks, metad


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
        assert len(queries) == 1
        raw_toks = self.mod_tokenizer.bert_tokenizer.tokenize(queries[0])
        raw_toks = [BERT_SOS.token_string] + raw_toks + [BERT_EOS.token_string]
        mod_toks, metad = self.mod_tokenizer.tokenize(queries[0])
        assert len(raw_toks) == len(mod_toks)
        indexed_tokens = self.mod_tokenizer.bert_tokenizer.convert_tokens_to_ids(raw_toks)
        segments_ids = [0 for _ in raw_toks]
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        #tokens_tensor = tokens_tensor.to('cuda')
        #segments_tensors = segments_tensors.to('cuda')
        last_layer, cls_value = self.bert(tokens_tensor, segments_tensors,
                                          output_all_encoded_layers=False)
        summaries = cls_value
        embedded_toks = last_layer

        return torch.Tensor(summaries), torch.Tensor(embedded_toks), mod_toks

    def get_tokenizer(self) -> ModStringTokenizerFromBert:
        return self._tokenizer


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
