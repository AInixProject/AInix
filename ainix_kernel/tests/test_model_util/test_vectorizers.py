from unittest.mock import MagicMock
from ainix_kernel.model_util.vectorizers import *

def test_avging():
    def x_vec_func(x):
        if x == "4":
            return torch.Tensor([4, 4, 4, 8, 8])
        else:
            return torch.Tensor([2, 2, 2, 2, 2])
    mockvocab = MagicMock(spec=Vocab)
    mockvocab.token_seq_to_indices = MagicMock(return_value=[1, 1])
    mockvocab.__len__ = lambda self: 3
    mocktokenizer = MagicMock()

    proc_func = make_xquery_avg_pretrain_func(mockvocab, mocktokenizer, x_vec_func)
    vectorizer = PretrainedAvgVectorizer(5, proc_func, mockvocab)

    pretrainer = vectorizer.get_pretrainer()
    mockast = MagicMock()
    mockexample = MagicMock()
    pretrainer.pretrain_example(mockexample, mockast)
    mockexample.xquery = "4"
    mockvocab.token_seq_to_indices = MagicMock(return_value=[0, 0])
    pretrainer.pretrain_example(mockexample, mockast)
    mockvocab.token_seq_to_indices = MagicMock(return_value=[2, 2])
    pretrainer.pretrain_example(mockexample, mockast)
    mockvocab.token_seq_to_indices = MagicMock(return_value=[1, 1])
    pretrainer.pretrain_example(mockexample, mockast)
    pretrainer.close()
    assert torch.all(vectorizer(torch.LongTensor([[0, 1, 2]])).eq(
           torch.Tensor([[[4, 4, 4, 8, 8],
                          [3, 3, 3, 5, 5],
                          [4, 4, 4, 8, 8]]])
    ))
    assert torch.all(vectorizer.counts.eq(torch.Tensor([2.0, 4.0, 2.0])))
