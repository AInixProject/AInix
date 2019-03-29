from typing import Tuple, List, Optional

import torch

from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode
from ainix_common.parsing.stringparser import AstUnparser, StringParser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import ExamplesStore
from ainix_kernel.model_util import vocab, vectorizers
from ainix_common.parsing.model_specific import tokenizers, parse_constants
from ainix_common.parsing.model_specific.tokenizers import NonLetterTokenizer, AstValTokenizer, \
    ModifiedWordPieceTokenizer, get_default_pieced_tokenizer_word_list
from ainix_kernel.model_util.glove import GloveIndex, get_glove_words
from ainix_kernel.model_util.vectorizers import set_deep_embed_from_glove
from ainix_kernel.model_util.vocab import Vocab, BasicVocab
from ainix_kernel.models.EncoderDecoder import encoders, decoders
from ainix_kernel.models.EncoderDecoder.decoders import TreeDecoder, TreeRNNDecoder
from ainix_kernel.models.EncoderDecoder.encoders import QueryEncoder, StringQueryEncoder, \
    RNNSeqEncoder, SimpleGloveEncoder
from ainix_kernel.models.LM.cookiemonster import make_default_cookie_monster_base, \
    PretrainPoweredQueryEncoder
from ainix_kernel.models.model_types import StringTypeTranslateCF, TypeTranslatePredictMetadata
from ainix_kernel.training.augmenting.replacers import Replacer, get_all_replacers
from ainix_kernel.training.model_specific_training import update_latent_store_from_examples


class EncDecModel(StringTypeTranslateCF):
    def __init__(
        self,
        type_context: TypeContext,
        query_encoder: StringQueryEncoder,
        tree_decoder: TreeDecoder
    ):
        self.type_context = type_context
        self.query_encoder = query_encoder
        self.decoder = tree_decoder
        self.modules = torch.nn.ModuleList([self.query_encoder, self.decoder])
        self.is_in_training_session = False
        self.optimizer: torch.optim.Optimizer = None

    def predict(
        self,
        x_string: str,
        y_type_name: str,
        use_only_train_data: bool
    ) -> Tuple[ObjectChoiceNode, TypeTranslatePredictMetadata]:
        if self.modules.training:
            raise ValueError("Should be in eval mode to predict")
        query_summary, encoded_tokens, actual_tokens = self.query_encoder([x_string])
        root_type = self.type_context.get_type_by_name(y_type_name)
        out_node, predict_data = self.decoder.forward_predict(
            query_summary, encoded_tokens, actual_tokens, root_type)
        out_node.freeze()
        return out_node, predict_data

    def train(
        self,
        x_string: str,
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode,
        example_id: int
    ) -> float:
        return self.train_batch([(x_string, y_ast, teacher_force_path, example_id)])

    def train_batch(
        self,
        batch: List[Tuple[str, AstObjectChoiceSet, ObjectChoiceNode, int]]
    ):
        if not self.is_in_training_session:
            raise ValueError("Call start_training_session before calling this.")
        if not self.modules.training:
            raise ValueError("Not in train mode. Call set_in_eval_mode")
        self.optimizer.zero_grad()
        xs, ys, teacher_force_paths, example_ids = zip(*batch)
        query_summary, encoded_tokens, actual_tokens = self.query_encoder(xs)
        loss = self.decoder.forward_train(
            query_summary, encoded_tokens, actual_tokens, ys, teacher_force_paths, example_ids)
        loss.backward()
        self.optimizer.step(None)
        return float(loss)

    #def train_only_decoder(self):
    #    self.optimizer.zero_grad()
    #    loss = self.decoder.forward_train(
    #        query_summary, encoded_tokens, actual_tokens, ys, teacher_force_paths, example_ids)
    #    self.optimizer.step(None)
    #    return float(loss)

    @classmethod
    def make_examples_store(cls, type_context: TypeContext, is_training: bool) -> ExamplesStore:
        raise NotImplemented

    def start_train_session(self):
        self.optimizer = torch.optim.Adam(self.modules.parameters())
        #self.optimizer = torch.optim.SGD(self.modules.parameters(), lr=0.01)
        self.decoder.start_train_session()
        self.set_in_train_mode()
        self.is_in_training_session = True

    def end_train_session(self):
        self.optimizer = None
        self.decoder.end_train_session()
        self.set_in_eval_mode()
        self.is_in_training_session = False

    def set_in_eval_mode(self):
        self.modules.eval()

    def set_in_train_mode(self):
        self.modules.train()

    def set_shared_memory(self):
        self.modules.share_memory()

    def get_string_tokenizer(self) -> tokenizers.StringTokenizer:
        return self.query_encoder.get_tokenizer()

    def get_latent_select_states(
        self,
        x_string: str,
        force_path: ObjectChoiceNode
    ) -> List[torch.Tensor]:
        query_summary, encoded_tokens, actual_tokens = self.query_encoder([x_string])
        return self.decoder.get_latent_select_states(
            query_summary, encoded_tokens, actual_tokens, force_path)

    def get_save_state_dict(self) -> dict:
        # TODO (DNGros): actually handle serialization rather than just letting pickling handle it
        # Custom handling has advantages in better handling stuff changed values on only parts of
        # the type contexts and vocabs and proper quick pretraining.
        return {
            "version": 0,
            'name': 'EncoderDecoder',
            "query_encoder": self.query_encoder.get_save_state_dict(),
            "decoder": self.decoder.get_save_state_dict(),
            # TODO figure out a better interface for this
            "need_latent_train": self.plz_train_this_latent_store_thanks() is not None,
            "need_example_store": self.plz_train_this_latent_store_thanks() is not None
        }

    @classmethod
    def create_from_save_state_dict(
        cls,
        state_dict: dict,
        new_type_context: TypeContext,
        new_example_store: ExamplesStore,
    ) -> 'EncDecModel':
        # TODO (DNGros): acutally handle the new type context.

        # TODO check the name of the query encoder
        query_encoder = PretrainPoweredQueryEncoder.create_from_save_state_dict(
            state_dict['query_encoder'])
        parser = StringParser(new_type_context)
        unparser = AstUnparser(new_type_context, query_encoder.get_tokenizer())
        replacers = get_all_replacers()
        decoder = TreeRNNDecoder.create_from_save_state_dict(state_dict[
            'decoder'], new_type_context, new_example_store, replacers, parser, unparser)
        model = cls(
            type_context=new_type_context,
            query_encoder=query_encoder,
            tree_decoder=decoder
        )
        if state_dict['need_latent_train']:
            update_latent_store_from_examples(
                model, decoder.action_selector.latent_store, new_example_store, replacers,
                parser, None, unparser, query_encoder.get_tokenizer()
            )
        return model

    def plz_train_this_latent_store_thanks(self):
        """Really crappy interface for getting the trainer to update the latent store at
        the end of an epoch because I want to sleep and annoyed and just want this to work
        and don't feel like designing a good interface."""
        return None


# Factory methods for different versions
def get_default_tokenizers() -> Tuple[
    Tuple[tokenizers.Tokenizer, Optional[Vocab]],
    tokenizers.Tokenizer
]:
    """Returns tuple (default x tokenizer, default y tokenizer)"""
    return (NonLetterTokenizer(), None), AstValTokenizer()
    word_piece_tok, word_list = get_default_pieced_tokenizer_word_list()
    x_vocab = BasicVocab(word_list + parse_constants.ALL_SPECIALS)
    return (word_piece_tok, x_vocab), AstValTokenizer()


def make_default_query_encoder(
    x_tokenizer: tokenizers.Tokenizer,
    query_vocab: Vocab,
    output_size=64,
    pretrain_checkpoint: str = None
) -> QueryEncoder:
    """Factory for making a default QueryEncoder"""
    #return SimpleGloveEncoder(query_vocab, 300)
    if pretrain_checkpoint is None:
        embed_size = output_size # 300
        x_vectorizer = vectorizers.TorchDeepEmbed(len(query_vocab), embed_size)
        print(f"encdecmodel:make_default_query_encoder {len(query_vocab)}")
        #set_deep_embed_from_glove(
        #    x_vectorizer, get_glove_words(embed_size, query_vocab), query_vocab)

        internal_encoder = RNNSeqEncoder(
            input_dims=x_vectorizer.feature_len(),
            hidden_size=output_size,
            summary_size=output_size,
            memory_tokens_size=output_size,
            variable_lengths=True,
            input_dropout_p=0.3,
            dropout_p=0.3,
            num_layers=1
        )
        return StringQueryEncoder(x_tokenizer, query_vocab, x_vectorizer, internal_encoder)
        #base_enc = make_default_cookie_monster_base(
        #    query_vocab, output_size, num_layers=1)
        #return PretrainPoweredQueryEncoder(
        #    x_tokenizer, query_vocab, base_enc, output_size,
        #    learned_extra_transform=True
        #)
    else:
        return PretrainPoweredQueryEncoder.create_with_pretrained_checkpoint(
            pretrain_checkpoint,
            x_tokenizer, query_vocab, output_size, freeze_base=True
        )


def get_default_encdec_model(
    examples: ExamplesStore,
    standard_size=16,
    replacer: Replacer = None,
    use_retrieval_decoder: bool = False,
    pretrain_checkpoint: str = None
):
    (x_tokenizer, x_vocab), y_tokenizer = get_default_tokenizers()
    if x_vocab is None or pretrain_checkpoint is None:
        # If we don't have a pretrained_checkpoint, then we might not know all
        # words in the vocab, so should filter the vocab to only tokens in dataset
        x_vocab = vocab.make_x_vocab_from_examples(
            examples, x_tokenizer, replacer, min_freq=2, replace_samples=2)
    hidden_size = standard_size
    tc = examples.type_context
    encoder = make_default_query_encoder(x_tokenizer, x_vocab, hidden_size, pretrain_checkpoint)
    if not use_retrieval_decoder:
        decoder = decoders.get_default_nonretrieval_decoder(tc, hidden_size)
    else:
        parser = StringParser(tc)
        unparser = AstUnparser(tc, x_tokenizer)
        decoder = decoders.get_default_retrieval_decoder(
            tc, hidden_size, examples, replacer, parser, unparser)
    model = EncDecModel(examples.type_context, encoder, decoder)
    if use_retrieval_decoder:
        # TODO lolz, this is such a crappy interface
        model.plz_train_this_latent_store_thanks = lambda: decoder.action_selector.latent_store
    return model

