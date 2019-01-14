from aish.strformat_utils import *


def test_get_highlighted_text():
    result = get_highlighted_text("0123456789", ((0, 4), (7, 9)), "S", "E", "!")
    assert result == "S0123E456S78E9!"


def test_get_highlighted_text2():
    result = get_highlighted_text("0123456789", ((3, 4), (7, 9)), "S", "E", "!")
    assert result == "E012S3E456S78E9!"


def test_get_only_text_in_intervals():
    result = get_only_text_in_intervals("0123456789", ((3, 4), (7, 9)), exclude_filler="-")
    assert result == "---3---78-"


def test_get_only_text_in_intervals2():
    result = get_highlighted_text("0123456789", ((0, 4), (7, 10)), excluded_exscape="-")
    assert result == "0123---789"
