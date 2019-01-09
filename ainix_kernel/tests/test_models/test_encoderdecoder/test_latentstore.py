from ainix_kernel.models.EncoderDecoder.latentstore import *
from ainix_kernel.tests.testutils.torch_test_utils import eps_eq_at, torch_epsilon_eq


def test_latent_store():
    instance = TorchLatentStore(
        type_ind_to_latents=[
            LatentDataWrapper.construct(
                latents=torch.Tensor([[1, 2], [3, 1], [-3, 1], [2, -1]]),
                example_indxs=torch.LongTensor([0, 1, 2, 3]),
                y_inds=torch.LongTensor([2, 2, 2, 2]),
                impl_choices=torch.LongTensor([3, 1, 9, 1])
            )
        ],
        example_to_depths_to_ind_in_types=None
    )
    data, similarities = instance.get_n_nearest_latents(torch.Tensor([1, 1]), 0)
    print(similarities)
    assert torch.all(data.example_indxs == torch.LongTensor([1, 0, 3, 2]))
    assert torch.all(data.impl_choices == torch.LongTensor([1, 3, 1, 9]))
    torch_epsilon_eq(similarities, torch.Tensor([4, 3, 1, -2]))

