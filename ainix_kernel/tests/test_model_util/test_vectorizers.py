from unittest.mock import MagicMock
from ainix_kernel.model_util.vectorizers import *

def test_avging():
    mockvocab = MagicMock()
    def x_vec_func(x):
        if x == "4":
            return torch.Tensor([4, 4, 4, 8, 8])
        else:
            return torch.Tensor([2, 2, 2, 2, 2])
    mockvocab.token_to_index = lambda x: 1
    mockvocab.__len__ = lambda self: 3

    proc_func = make_xquery_avg_pretrain_func(mockvocab, x_vec_func)
    vectorizer = PretrainedAvgVectorizer(5, proc_func, mockvocab)

    pretrainer = vectorizer.get_pretrainer()
    mockast = MagicMock()
    mockexample = MagicMock()
    mockast.depth_first_iter = lambda: ["a", "b"]
    pretrainer.pretrain_example(mockexample, mockast)
    mockexample.xquery = "4"
    mockvocab.token_to_index = lambda x: 0
    pretrainer.pretrain_example(mockexample, mockast)
    mockvocab.token_to_index = lambda x: 2
    pretrainer.pretrain_example(mockexample, mockast)
    mockvocab.token_to_index = lambda x: 1
    pretrainer.pretrain_example(mockexample, mockast)
    pretrainer.close()
    assert torch.all(vectorizer(torch.LongTensor([[0, 1, 2]])).eq(
           torch.Tensor([[[4, 4, 4, 8, 8],
                          [3, 3, 3, 5, 5],
                          [4, 4, 4, 8, 8]]])
    ))
    assert torch.all(vectorizer.counts.eq(torch.Tensor([2, 4, 2])))
