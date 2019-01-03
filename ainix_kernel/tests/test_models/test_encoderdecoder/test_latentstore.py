from ainix_kernel.models.EncoderDecoder.latentstore import *
from ainix_kernel.tests.testutils.torch_test_utils import eps_eq_at, torch_epsilon_eq


def test_single_store():
    instance = SingleTypeLatentStore(2)
    instance.add_latent(torch.Tensor([1,2]), 0, 2, 3)
    instance.add_latent(torch.Tensor([3,1]), 1, 2, 1)
    instance.add_latent(torch.Tensor([-3,1]), 2, 2, 9)
    instance.add_latent(torch.Tensor([2,-1]), 3, 2, 1)
    instance.set_read()

    metadata, similarities, v = instance.get_n_nearest_latents(torch.Tensor([1,1]), 10)
    print(similarities)
    assert torch.all(metadata.example_indxs == torch.LongTensor([1,0,3,2]))
    assert torch.all(metadata.impl_choices == torch.LongTensor([1,3,1,9]))
    torch_epsilon_eq(similarities, torch.Tensor([4, 3, 1, -2]))