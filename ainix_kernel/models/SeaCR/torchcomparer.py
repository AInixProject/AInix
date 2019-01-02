from ainix_kernel.models.SeaCR.comparer import Comparer, SimpleRulebasedComparer
from abc import abstractmethod
from ainix_common.parsing.model_specific.tokenizers import Tokenizer
from ainix_kernel.model_util.transformer.layers import EncoderLayer
from ainix_kernel.model_util.transformer.sublayers import MultiHeadAttention
from ainix_kernel.model_util.vectorizers import VectorizerBase, TorchDeepEmbed
from ainix_kernel.model_util.vocab import Vocab
from ainix_kernel.models.SeaCR import treeutil
from ainix_kernel.models.SeaCR.comparer import ComparerResult
from ainix_common.parsing.ast_components import ObjectChoiceNode, AstNode
import attr
import torch
from torch import nn
from ainix_kernel.model_util.modelcomponents import TimingSignalAdd, EmbeddedAdder


class FieldComparerPredictor(torch.nn.Module):
    @abstractmethod
    def forward(self, vectors_gen_query, vectors_gen_ast,
                vectors_example_query, vectors_example_ast):
        pass

class TransformerPredictor(FieldComparerPredictor):
    def __init__(self, hidden_size: int, num_layers, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(*[
            EncoderLayer(
                hidden_size=self.hidden_size,
                total_key_depth=self.hidden_size,
                total_value_depth=self.hidden_size,
                filter_size=self.hidden_size,
                num_heads=num_heads
            )
            for _ in range(num_layers)]
        )
        # Here we define a series of special keys/values which we look at
        # while getting our result.
        self.out_prob_key = torch.nn.Parameter(torch.randn(hidden_size))
        self.out_select_key = torch.nn.Parameter(torch.randn(hidden_size))
        self.special_keys = torch.stack((self.out_prob_key, self.out_select_key))
        self.specials_length = self.special_keys.shape[0]
        assert self.specials_length == 2
        assert self.special_keys.shape[1] == self.hidden_size

        self.specials_model = MultiHeadAttention(
            input_depth=self.hidden_size,
            total_key_depth=self.hidden_size,
            total_value_depth=self.hidden_size,
            output_depth=self.hidden_size,
            num_heads=1
        )

        self.out_prob_linear_hidden = nn.Linear(self.hidden_size, int(self.hidden_size / 2))
        self.out_prob_nonlin = nn.ReLU()
        self.out_prob_linear = nn.Linear(int(self.hidden_size / 2), 1)

    def forward(self, vectors_gen_query, vectors_gen_ast,
                vectors_example_query, vectors_example_ast):
        # Concat all our input fields into one big tensor for input into transformers
        num_of_batches = vectors_example_ast.shape[0]
        expanded_specials = self.special_keys.expand(num_of_batches, -1, -1)
        combined_vector = torch.cat(
            (vectors_gen_query, vectors_example_query,
             vectors_example_query, vectors_example_ast,
             expanded_specials),
            dim=1
        )
        # Pass everything through transformers
        encoded = self.encoder(combined_vector)
        # Pull out the special and non special keys
        encoded_combined = encoded[:, :-self.specials_length, :]
        encoded_specials = encoded[:, -self.specials_length:, :]
        # Pass the specials through one extra transformer
        specials_extra_enc = self.specials_model.forward(
            encoded_specials, encoded_combined, encoded_combined)
        # Look at the specials in order to determine outputs
        out_prob_vec = specials_extra_enc[:, 0]
        out_prob_score = self.out_prob_linear(
            self.out_prob_nonlin(self.out_prob_linear_hidden(out_prob_vec)))
        return out_prob_score

class FirstWordPredictor(FieldComparerPredictor):
    """A stupid field predictor used for quick test."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidlinear = nn.Linear(hidden_size*2, hidden_size)
        self.nonlin = nn.ReLU()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, vectors_gen_query, vectors_gen_ast,
                vectors_example_query, vectors_example_ast):
        first_word_gen = vectors_gen_query[:, 0]
        first_word_example = vectors_example_query[:, 0]
        combined = torch.cat((first_word_gen, first_word_example), dim=1)
        #combined = first_word_gen - first_word_example
        #print("first_word_gen", first_word_gen)
        #print("first_word_example", first_word_example)
        #print("combined", combined)
        #print("linear", self.linear.weight)
        return self.linear(self.nonlin(self.hidlinear(combined)))


@attr.s(auto_attribs=True)
class TorchComparer(Comparer):
    """A TorchComparer is uses PyTorch to learn to create a ComparerResult.
    It takes in (quite a lot) of inputs, to allow one to create new comparers
    by composing different tokenizers, vectorizers, and field predictors.

    The basic pipeline is it takes in four values
    (query we are generating for, the ast root we have so far,
     the query we are comparing, the ast root we are comparing)
    Then for each of those inputs it goes:
    input -> tokenize with tokenizer -> convert to indicies with vocab ->
             pass through appriate vectorizer -> pass each set of vectors into
             FieldComparePredictor to predict result.
    """
    x_vocab: Vocab
    y_vocab: Vocab
    x_tokenizer: Tokenizer
    y_tokenizer: Tokenizer
    gen_query_vectorizer: VectorizerBase
    example_query_vectorizer: VectorizerBase
    gen_ast_vectorizer: VectorizerBase
    example_ast_vectorizer: VectorizerBase
    fields_compare_predictor: FieldComparerPredictor

    def __attrs_post_init__(self):
        self.all_models = torch.nn.ModuleList(
            [self.gen_query_vectorizer, self.example_query_vectorizer,
             self.gen_ast_vectorizer, self.example_ast_vectorizer,
             self.fields_compare_predictor]
        )

    def _compare_internal(
        self,
        gen_query: str,
        gen_ast_current_root: ObjectChoiceNode,
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: AstNode,
    ):
        #print("---")
        #print("gen_query", gen_query)
        #print("example_query", example_query)

        # TODO (DNGros): cache the gen stuff somehow. This will likely come along
        # with implementing a batched comparer
        # First let's tokenize all our inputs
        tokenized_gen_query, _ = self.x_tokenizer.tokenize(gen_query)
        tokenized_gen_ast, gen_node_pointers = self.y_tokenizer.tokenize(gen_ast_current_root)
        tokenized_example_query, _ = self.x_tokenizer.tokenize(example_query)
        tokenized_example_ast, example_node_pointers = self.y_tokenizer.tokenize(example_ast_root)

        #print("tokenized_gen_query", tokenized_gen_query)
        #print("tokenized_example_query", tokenized_example_query)

        # Convert our tokens into indices
        indices_gen_query = self.x_vocab.token_seq_to_indices([tokenized_gen_query])
        indices_gen_ast = self.y_vocab.token_seq_to_indices([tokenized_gen_ast])
        indices_example_query = self.x_vocab.token_seq_to_indices([tokenized_example_query])
        indices_example_ast = self.y_vocab.token_seq_to_indices([tokenized_example_ast])

        # Convert them to vectors
        vectors_gen_query = self.gen_query_vectorizer(indices_gen_query)
        vectors_gen_ast = self.gen_ast_vectorizer(indices_gen_ast)
        vectors_example_query = self.example_query_vectorizer(indices_example_query)
        vectors_example_ast = self.example_ast_vectorizer(indices_example_ast)

        #print("vectors_gen_query", vectors_gen_query)
        #print("vectors_example_query", vectors_example_query)

        # Get result
        out_prob_score = self.fields_compare_predictor.forward(
            vectors_gen_query, vectors_gen_ast, vectors_example_query, vectors_example_ast
        )
        return out_prob_score

    def compare(
        self,
        gen_query: str,  # maybe we should just pass in the already tokenized version
        gen_ast_current_root: ObjectChoiceNode,
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: AstNode,
    ) -> ComparerResult:
        # TODO (DNGros): It would be nice to also pass in other features into
        # this such as the root type of the example vs the root type of the gen
        out_prob_score = self._compare_internal(
            gen_query,
            gen_ast_current_root,
            gen_ast_current_leaf,
            current_gen_depth,
            example_query,
            example_ast_root,
        )
        out_prob_logits = torch.nn.functional.sigmoid(out_prob_score)
        # for now assume just one in
        out_prob_logit = out_prob_logits[0]
        # just use the rulebased thing for selecting which one for now
        potential_type_choice_nodes = treeutil.get_type_choice_nodes(
            example_ast_root, gen_ast_current_leaf.get_type_to_choose_name())
        depth_diffs = SimpleRulebasedComparer.get_impl_depth_difference(
            potential_type_choice_nodes, current_gen_depth)
        ranked_options = sorted(
            [(-score, name) for name, score in depth_diffs.items()],
            reverse=True
        )
        return ComparerResult(out_prob_logits, tuple(ranked_options))

    def train(
        self,
        gen_query: str,
        gen_ast_current_root: ObjectChoiceNode,
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: AstNode,
        expected_result: ComparerResult
    ) -> torch.Tensor:
        out_prob_score = self._compare_internal(
            gen_query,
            gen_ast_current_root,
            gen_ast_current_leaf,
            current_gen_depth,
            example_query,
            example_ast_root,
        )
        return out_prob_score

    def get_parameters(self):
        #print(list(self.all_models.parameters()))
        return self.all_models.parameters()


def get_default_torch_comparer(x_vocab, y_vocab, x_tokenizer, y_tokenizer, out_dims=8):
    x_embed = TorchDeepEmbed(len(x_vocab), out_dims)
    y_embed = TorchDeepEmbed(len(x_vocab), out_dims)
    return TorchComparer(
        x_vocab=x_vocab,
        y_vocab=y_vocab,
        x_tokenizer=x_tokenizer,
        y_tokenizer=y_tokenizer,
        gen_query_vectorizer=SimpleQueryVectorizer(out_dims, x_embed),
        example_query_vectorizer=SimpleQueryVectorizer(out_dims, x_embed),
        gen_ast_vectorizer=SimpleQueryVectorizer(out_dims, y_embed),
        example_ast_vectorizer=SimpleQueryVectorizer(out_dims, y_embed),
        fields_compare_predictor=TransformerPredictor(out_dims, 2, num_heads=4)
        #fields_compare_predictor=FirstWordPredictor(out_dims)
    )
    pass


class SimpleQueryVectorizer(VectorizerBase):
    def __init__(self, out_dims, embed: TorchDeepEmbed):
        super().__init__()
        self.out_dims = out_dims
        self.embed = embed
        # Add in some signal for what field this is
        self.field_adder = EmbeddedAdder(out_dims)
        self.timing_signal = TimingSignalAdd()

    def feature_len(self):
        return self.out_dims

    def forward(self, indices: torch.Tensor):
        return self.timing_signal(self.field_adder(self.embed(indices)))


# TODO (DNGros): Batched version of comparers
# A batch comparer could probably take the form of where it takes in lists of
# gen values, and example values. Then it also takes in lists of indicies which
# select into the list of gen values, and example values. This could for example
# all you to have one gen value which is reused in comparing maybe ten examples.
