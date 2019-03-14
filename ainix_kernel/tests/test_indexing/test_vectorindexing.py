from ainix_kernel.indexing.vectorindexing import *
from ainix_kernel.tests.testutils.torch_test_utils import torch_epsilon_eq


def test_torchvectdb():
    builder = TorchVectorDatabaseBuilder(
        key_dimensionality=2,
        value_fields=[VDBIntField('foo'), VDBStringField('bar')]
    )
    builder.add_data(np.array([1., 3]), (2, 'hi'))
    builder.add_data(np.array([1., -3]), (5, 'yo'))
    builder.add_data(np.array([-1., 1]), (7, 'jo'))

    db = builder.produce_result()
    values, sims = db.get_n_nearest(torch.tensor([4., -1]), max_results=2)
    assert values == [(5, 'yo'), (2, 'hi')]
    assert torch_epsilon_eq(sims, [7., 1])

    nearest, sim = db.get_nearest(torch.tensor([4., -1]))
    assert nearest == (5, 'yo')
    assert sim == torch.tensor(7.)
