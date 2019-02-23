from builtin_types.otherdata.tellina.convert_data import merge_command_sets


def test_merge1():
    merged = merge_command_sets([("x", "y"), ("xx", "yy"), ("z", "y")])
    print(merged)
    assert merged == [(["x", 'z'], ['y']), (["xx"], ["yy"])]


def test_merge2():
    merged = merge_command_sets([("x", "y"), ("xx", "yy"), ("z", "y"), ("a", "b"), ("a", "y")])
    assert merged == [(['a', "x", 'z'], ['b', 'y']), (["xx"], ["yy"])]
