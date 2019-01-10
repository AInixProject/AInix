from typing import Tuple, List

import torch

from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import ExamplesStore
from ainix_kernel.model_util import vocab, vectorizers
from ainix_common.parsing.model_specific import tokenizers
from ainix_common.parsing.model_specific.tokenizers import NonLetterTokenizer, AstValTokenizer
from ainix_kernel.models.EncoderDecoder import encoders, decoders
from ainix_kernel.models.EncoderDecoder.decoders import TreeDecoder, TreeRNNDecoder
from ainix_kernel.models.EncoderDecoder.encoders import QueryEncoder, StringQueryEncoder
from ainix_kernel.models.model_types import StringTypeTranslateCF
from ainix_kernel.training.augmenting.replacers import Replacer


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
    ) -> ObjectChoiceNode:
        if self.modules.training:
            raise ValueError("Should be in eval mode to predict")
        query_summary, encoded_tokens = self.query_encoder([x_string])
        root_type = self.type_context.get_type_by_name(y_type_name)
        out_node = self.decoder.forward_predict(query_summary, encoded_tokens, root_type)
        out_node.freeze()
        return out_node

    def train(
        self,
        x_string: str,
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode,
        example_id: int
    ) -> torch.Tensor:
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
        query_summary, encoded_tokens = self.query_encoder(xs)
        loss = self.decoder.forward_train(
            query_summary, encoded_tokens, ys, teacher_force_paths, example_ids)
        print("SDFSDFSDF LOSS 🦔", loss)
        loss.backward()
        self.optimizer.step(None)
        return loss

    @classmethod
    def make_examples_store(cls, type_context: TypeContext, is_training: bool) -> ExamplesStore:
        raise NotImplemented

    def start_train_session(self):
        self.optimizer = torch.optim.Adam(self.modules.parameters())
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
        query_summary, encoded_tokens = self.query_encoder([x_string])
        return self.decoder.get_latent_select_states(query_summary, encoded_tokens, force_path)

    def get_save_state_dict(self) -> dict:
        # TODO (DNGros): actually handle serialization rather than just letting pickling handle it
        # Custom handling has advantages in better handling stuff changed values on only parts of
        # the type contexts and vocabs and proper quick pretraining.
        return {
            "version": 0,
            "query_encoder": self.query_encoder.get_save_state_dict(),
            "decoder": self.decoder.get_save_state_dict()
        }

    @classmethod
    def create_from_save_state_dict(
        cls,
        state_dict: dict,
        new_type_context: TypeContext,
        new_example_store: ExamplesStore
    ) -> 'EncDecModel':
        # TODO (DNGros): acutally handle the new type context.
        query_encoder = StringQueryEncoder.create_from_save_state_dict(state_dict['query_encoder'])
        decoder = TreeRNNDecoder.create_from_save_state_dict(
            state_dict['decoder'], new_type_context, new_example_store)
        return cls(
            type_context=new_type_context,
            query_encoder=query_encoder,
            tree_decoder=decoder
        )

    def plz_train_this_latent_store_thanks(self):
        """Really crappy interface for getting the trainer to update the latent store at
        the end of an epoch because I want to sleep and annoyed and just want this to work
        and don't feel like designing a good interface."""
        return None


# Factory methods for different versions
def _get_default_tokenizers() -> Tuple[tokenizers.Tokenizer, tokenizers.Tokenizer]:
    """Returns tuple (default x tokenizer, default y tokenizer)"""
    return NonLetterTokenizer(), AstValTokenizer()


def get_default_encdec_model(examples: ExamplesStore, standard_size=16, replacer: Replacer = None,
                             use_retrieval_decoder: bool = False):
    x_tokenizer, y_tokenizer = _get_default_tokenizers()
    x_vocab = vocab.make_x_vocab_from_examples(examples, x_tokenizer)
    hidden_size = standard_size
    tc = examples.type_context
    encoder = encoders.make_default_query_encoder(x_tokenizer, x_vocab, hidden_size)
    if not use_retrieval_decoder:
        decoder = decoders.get_default_nonretrieval_decoder(tc, hidden_size)
    else:
        decoder = decoders.get_default_retrieval_decoder(tc, hidden_size, examples, replacer)
    model = EncDecModel(examples.type_context, encoder, decoder)
    if use_retrieval_decoder:
        # TODO lolz, this is such a crappy interface
        model.plz_train_this_latent_store_thanks = lambda s: decoder.action_selector.latent_store
    return model

