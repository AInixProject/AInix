from typing import Tuple, Optional

from ainix_common.parsing import copy_tools
from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_common.parsing.model_specific.tokenizers import StringTokenizer
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_kernel.indexing.examplestore import ExamplesStore, DataSplits
from ainix_kernel.models.EncoderDecoder.latentstore import LatentStore
from ainix_kernel.models.model_types import StringTypeTranslateCF
from ainix_kernel.training.augmenting.replacers import Replacer, seed_from_x_val


def update_latent_store_from_examples(
    model: 'StringTypeTranslateCF',
    latent_store: LatentStore,
    examples: ExamplesStore,
    replacer: Replacer,
    parser: StringParser,
    splits: Optional[Tuple[DataSplits]],
    unparser: AstUnparser,
    tokenizer: StringTokenizer
):
    model.set_in_eval_mode()
    for example in examples.get_all_x_values(splits):
        # TODO multi sampling and average replacers
        x_replaced, y_replaced = replacer.strings_replace(
            example.xquery, example.ytext, seed_from_x_val(example.xquery))
        ast = parser.create_parse_tree(y_replaced, example.ytype)
        _, token_metadata = tokenizer.tokenize(x_replaced)
        copy_ast = copy_tools.make_copy_version_of_tree(ast, unparser, token_metadata)
        # TODO Think about whether feeding in the raw x is good idea.
        # will change once have replacer sampling
        latents = model.get_latent_select_states(example.xquery, copy_ast)
        nodes = list(copy_ast.depth_first_iter())
        #print("LATENTS", latents)
        for i, l in enumerate(latents):
            dfs_depth = i*2
            n = nodes[dfs_depth].cur_node
            assert isinstance(n, ObjectChoiceNode)
            c = l.detach()
            assert not c.requires_grad
            latent_store.set_latent_for_example(c, n.type_to_choose.ind,
                                                example.id, dfs_depth)
    model.set_in_train_mode()