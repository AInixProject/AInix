from ainix_kernel.model_util.stringops import *
from ainix_kernel.tests.testutils.torch_test_utils import torch_epsilon_eq
from ainix_common.parsing.model_specific.parse_constants import PAD


def mtok(val: str, after_whitespace: bool = True) -> ModifiedStringToken:
     return ModifiedStringToken(val, CasingModifier.LOWER,
                                WhitespaceModifier.AFTER_SPACE_OR_SOS if after_whitespace else
                                WhitespaceModifier.NOT_AFTER_SPACE)


def test_word_lens_tokens():
     v = get_word_lens_of_moded_tokens(
          [[mtok("my"), mtok("nme"), mtok("'s", False), mtok("s"), mtok("o", False),
            mtok("p", False), mtok(".")]]
     )
     assert torch_epsilon_eq(
         v,
         [[1, 2, 2, 3, 3, 3, 1]]
     )


def test_word_lens_tokens2():
    v = get_word_lens_of_moded_tokens(
        [[mtok("my"), mtok("nme"), mtok("'s", False)]]
    )
    assert torch_epsilon_eq(
        v,
        [[1, 2, 2]]
    )


def test_word_lens_tokens_with_pad():
    v = get_word_lens_of_moded_tokens([
        [mtok("my"), mtok("nme"), mtok("'s", False), mtok(PAD)],
        [mtok("my"), mtok("nme"), mtok("'s", False), mtok(".")],
        [mtok("my"), mtok("nme"), mtok(PAD), mtok(PAD)]
    ])
    assert torch_epsilon_eq(
        v,
        [
            [1, 2, 2, 0],
            [1, 2, 2, 1],
            [1, 1, 0, 0]
        ]
    )


def test_word_lens_tokens_with_pad2():
    v = get_word_lens_of_moded_tokens([
        [mtok("my"), mtok("nme"), mtok("'s", False)],
        [mtok("my"), mtok("nme"), mtok("'s", False), mtok(".")],
        [mtok("my"), mtok("nme")]
    ])
    assert torch_epsilon_eq(
        v,
        [
            [1, 2, 2, 0],
            [1, 2, 2, 1],
            [1, 1, 0, 0]
        ]
    )
