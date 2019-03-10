import numpy as np

import torch

from sklearn.naive_bayes import GaussianNB

from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.stringparser import AstUnparser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import DataSplits
from ainix_kernel.models.Fullretrieval.fullretmodel import full_ret_from_example_store
from ainix_kernel.specialtypes import allspecials
from ainix_kernel.training.augmenting.replacers import get_all_replacers
from ainix_kernel.training.train_contexts import ALL_EXAMPLE_NAMES, load_all_examples

if __name__ == "__main__":
    pretrained_checkpoint_path = "../../checkpoints/" \
                                 "lmchkp_30epoch2rnn_merge_toks_total_2.922_ns0.424_lm2.4973.pt"

    type_context = TypeContext()
    loader = TypeContextDataLoader(type_context, up_search_limit=4)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml")
    loader.load_path("builtin_types/command.ainix.yaml")
    loader.load_path("builtin_types/paths.ainix.yaml")
    allspecials.load_all_special_types(type_context)

    for f in ALL_EXAMPLE_NAMES:
        loader.load_path(f"builtin_types/{f}.ainix.yaml")
    type_context.finalize_data()

    index = load_all_examples(type_context)
    #index = load_tellia_examples(type_context)

    print("num docs", index.get_doc_count())
    print("num train", len(list(index.get_all_examples((DataSplits.TRAIN, )))))

    replacers = get_all_replacers()

    model = full_ret_from_example_store(index, replacers, pretrained_checkpoint_path)
    unparser = AstUnparser(type_context, model.get_string_tokenizer())
    nb_models = model.nb_models
    program_nb = nb_models['Program']

    print(program_nb._model.sigma_)
    print(program_nb._model.theta_)

    #program_nb._model.sigma_ *= 100

    while True:
        q = input("Query: ")
        summary, mem, tokens = model.embedder([q])
        summary_and_depth = torch.cat(
            (summary, torch.tensor([[2.0]])), dim=1)
        s1 = summary_and_depth[0].data.numpy()
        std_devs = np.sqrt(program_nb._model.sigma_)
        diffs_from_mean = program_nb._model.theta_ - s1
        num_std_devs_diff = diffs_from_mean / std_devs
        variance = program_nb._model.theta_
        exp_power = -(diffs_from_mean**2 / (2*variance))
        exp_term = np.exp(exp_power)
        scaleing = (1/np.sqrt(2*np.pi*variance))
        density = scaleing * exp_term
        #print(program_nb._model.predict(summary.data.numpy()))
        #pred = program_nb._model.predict_proba(summary_and_depth.data.numpy())
        pred = program_nb.logit_model.predict_proba(summary_and_depth.data.numpy())

        print([f'{program_nb._ind_to_obj_name[i]}: {p:.3f}' for i, p in enumerate(pred[0])])
