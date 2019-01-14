from ainix_kernel.explan_tools.example_explan import _narrow_down_examples
from ainix_kernel.models.model_types import ExampleRetrieveExplanation


def test_narrow_down_examples():
    retr_explans = (
        ExampleRetrieveExplanation(
            reference_example_ids=(10, 4),
            reference_confidence=(1, 0),
            reference_example_dfs_ind=(1, 5)
        ),
        ExampleRetrieveExplanation(
            reference_example_ids=(3, 9),
            reference_confidence=(1, 0),
            reference_example_dfs_ind=(2, 5)
        ),
        ExampleRetrieveExplanation(
            reference_example_ids=(10, 8),
            reference_confidence=(1, 0),
            reference_example_dfs_ind=(3, 5)
        )
    )
    res = _narrow_down_examples(retr_explans)
    expected = [
        (10, (0, 4)),
        (3, (2,))
    ]
    print(res)
    assert res == expected
