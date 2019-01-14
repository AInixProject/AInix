"""Tools to help with post processing of ExampleRetrievalExplanations

TODO: Refactor this into ainix_common
"""
from typing import Tuple, List

from ainix_kernel.indexing.examplestore import ExamplesStore
from ainix_kernel.models.model_types import ExampleRetrieveExplanation


def post_process_explanations(
    retr_explans: Tuple[ExampleRetrieveExplanation, ...],
    example_store: ExamplesStore
) -> List[Tuple[str, str]]:
    out = []
    all_first_choices = [expl.reference_example_ids[0] for expl in retr_explans]
    all_first_choices_set = set(all_first_choices)
    for examp_id in all_first_choices_set:
        example = example_store.get_example_by_id(examp_id)
        out.append((example.xquery, example.ytext))
    return out
