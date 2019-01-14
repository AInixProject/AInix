"""Tools to help with post processing of ExampleRetrievalExplanations

TODO: Refactor this into ainix_common
"""
import numpy as np
from typing import Tuple, List, Sequence

from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_common.parsing.stringparser import StringParser, UnparseResult
from ainix_kernel.indexing.examplestore import ExamplesStore, Example
from ainix_kernel.models.model_types import ExampleRetrieveExplanation
import attr
from intervaltree import Interval, IntervalTree


@attr.s(auto_attribs=True, frozen=True)
class ExampleExplanPostProcessedOutput:
    example_str: str
    example_cmd: str
    input_str_intervals: List[Tuple[int, int]]


def post_process_explanations(
    retr_explans: Tuple[ExampleRetrieveExplanation, ...],
    example_store: ExamplesStore,
    outputted_ast: ObjectChoiceNode,
    outputted_unparser: UnparseResult
) -> List[ExampleExplanPostProcessedOutput]:
    narrowed_down_examples = _narrow_down_examples(retr_explans)
    out = []
    for example_id, use_dfs_ids in narrowed_down_examples:
        actual_example = example_store.get_example_by_id(example_id)
        intervals = _get_unparse_intervals_of_inds(use_dfs_ids, outputted_ast, outputted_unparser)
        if len(intervals) == 0:
            continue
        out.append(ExampleExplanPostProcessedOutput(
            example_str=actual_example.xquery,
            example_cmd=actual_example.ytext,
            input_str_intervals=_interval_tree_to_tuples(intervals)
        ))
    out.sort(key=lambda v: v.input_str_intervals[0][0])
    return out


def _narrow_down_examples(
    retr_explans: Tuple[ExampleRetrieveExplanation, ...]
) -> List[Tuple[int, Tuple[int]]]:
    """Figures out what examples we are actually using for the explanation. This
    will remove duplicates where one example is used for more than one part
    of the output.

    Returns:
        List of tuples:
            example_id: An example id we are going to use
            dfs_inds: the indexes into a dfs view of our model output that this
                example explains
    """
    all_first_choices = np.array([expl.reference_example_ids[0] for expl in retr_explans])
    all_first_choices_set = set(all_first_choices)
    # build the output. Note the multiply by two since we only get reference_example_ids for
    # the object choice nodes. Multiplly by two gets the proper depth with the ObjectNodes.
    return [(eid, tuple(np.where(all_first_choices == eid)[0] * 2))
            for eid in all_first_choices_set]


def _example_inds_to_examples(example_inds: Sequence[int], example_store: ExamplesStore):
    return [example_store.get_example_by_id(eid) for eid in example_inds]


def _interval_tree_to_tuples(interval_tree: IntervalTree) -> List[Tuple[int, int]]:
    return [(interval.begin, interval.end) for interval in interval_tree]


def _get_unparse_intervals_of_inds(
    dfs_inds_to_include: Sequence[int],
    ast: ObjectChoiceNode,
    unparse: UnparseResult
) -> IntervalTree:
    """Given some indicies we wish include, find the intervals of the total
    unparse string which are covered by those indicies"""
    include_set = set(dfs_inds_to_include)
    interval_tree = IntervalTree()
    currently_including = False
    for ind, pointer in enumerate(ast.depth_first_iter()):
        if ind % 2 != 0:
            # Only take into account the choice nodes. Skip the object nodes
            continue
        assert isinstance(pointer.cur_node, ObjectChoiceNode)
        func_need_to_do_here = None
        if ind in include_set:
            if not currently_including:
                func_need_to_do_here = lambda start, end: interval_tree.add(Interval(start, end))
                currently_including = True
        else:
            if currently_including:
                func_need_to_do_here = lambda start, end: interval_tree.chop(start, end)
                currently_including = False
        if func_need_to_do_here:
            span = unparse.pointer_to_span(pointer)
            if span is None or span[1] - span[0] == 0:
                continue
            start, end = span
            func_need_to_do_here(start, end)

    return interval_tree




