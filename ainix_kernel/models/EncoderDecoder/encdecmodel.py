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
        query_summary, encoded_tokens = self.query_encoder([x_string])
        root_type = self.type_context.get_type_by_name(y_type_name)
        out_node = self.decoder.forward_predict(query_summary, encoded_tokens, root_type)
        out_node.freeze()
        return out_node

    def train(
        self,
        x_string: str,
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode
    ) -> torch.Tensor:
        return self.train_batch([(x_string, y_ast, teacher_force_path)])

    def train_batch(
        self,
        batch: List[Tuple[str, AstObjectChoiceSet, ObjectChoiceNode]]
    ):
        if not self.is_in_training_session:
            raise ValueError("Call start_training_session before calling this.")
        self.optimizer.zero_grad()
        xs, ys, teacher_force_paths = zip(*batch)
        query_summary, encoded_tokens = self.query_encoder(xs)
        loss = self.decoder.forward_train(
            query_summary, encoded_tokens, ys, teacher_force_paths)
        loss.backward()
        self.optimizer.step(None)
        return loss

    @classmethod
    def make_examples_store(cls, type_context: TypeContext, is_training: bool) -> ExamplesStore:
        raise NotImplemented

    def start_train_session(self):
        self.modules.train()
        self.optimizer = torch.optim.Adam(self.modules.parameters())
        self.is_in_training_session = True

    def end_train_session(self):
        self.optimizer = None
        self.modules.eval()
        self.is_in_training_session = False

    def set_shared_memory(self):
        self.modules.share_memory()

    def get_string_tokenizer(self) -> tokenizers.StringTokenizer:
        return self.query_encoder.get_tokenizer()

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


# Factory methods for different versions
def _get_default_tokenizers() -> Tuple[tokenizers.Tokenizer, tokenizers.Tokenizer]:
    """Returns tuple (default x tokenizer, default y tokenizer)"""
    return NonLetterTokenizer(), AstValTokenizer()


def get_default_encdec_model(examples: ExamplesStore, standard_size=16):
    x_tokenizer, y_tokenizer = _get_default_tokenizers()
    x_vocab, y_vocab = vocab.make_vocab_from_example_store_and_type_context(examples, x_tokenizer)
    hidden_size = standard_size
    y_vectorizer = vectorizers.TorchDeepEmbed(len(y_vocab), hidden_size)
    encoder = encoders.make_default_query_encoder(x_tokenizer, x_vocab, hidden_size)
    decoder = decoders.get_default_decoder(y_vocab, y_vectorizer, hidden_size)
    return EncDecModel(examples.type_context, encoder, decoder)
