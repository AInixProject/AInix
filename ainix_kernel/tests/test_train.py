"""Attempts to unit test the various aspects the model
should be able to learn"""
import pytest
import pudb
import train
from ainix_kernel.program_description import AIProgramDescription, Argument
from ainix_kernel.bashmetrics import BashMetric
import random
from ainix_kernel import serialize_tools

def test_pick_program():
    """A test to see if can correctly learn to predict a program"""
    cow = AIProgramDescription(
        name = "cow"
    )
    dog = AIProgramDescription(
        name = "dog"
    )
    kitty = AIProgramDescription( # origionally tried cat, but....
        name = "kitty"
    )
    # make some toy data. It is multiplied by a large number to ensure
    # everything will make it into both train and val
    data = [
        ("go moo", "cow"),
        ("go meow", "kitty"),
        ("please go woof", "dog")
    ] * 100
    random.shuffle(data)
    # Do training
    train_output = train.run_with_data_list(data, [cow, dog, kitty], False, quiet_mode = True, num_epochs = 3)
    meta_model, final_state, train_iter, val_iter = train_output
    
    # eval the model. Expect should get basically perfect progam picking
    bashmetric = BashMetric()
    train.eval_model(meta_model, val_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.first_cmd_acc() >= 0.98

def test_pick_arg():
    """A test to see if can correctly learn to predict arg existance"""
    aArg = Argument("a", "StoreTrue")
    bArg = Argument("b", "StoreTrue")
    cow = AIProgramDescription(
        name = "cow",
        arguments = [aArg, bArg]
    )
    # make some toy data. It is multiplied by a large number to ensure
    # everything will make it into both train and val
    data = [
        ("give cow an apple", "cow -a"),
        ("give the cow a banana", "cow -b"),
        ("cow feast", "cow -a -b")
    ] * 100
    random.shuffle(data)
    # Do training
    train_output = train.run_with_data_list(data, [cow], False, quiet_mode = True, num_epochs = 5)
    meta_model, final_state, train_iter, val_iter = train_output
    
    # eval the model. Expect should get basically perfect progam picking
    bashmetric = BashMetric()
    train.eval_model(meta_model, val_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98

def test_fill_pos():
    posArg = Argument("aposarg", "Stringlike", position = 0)
    cow = AIProgramDescription(
        name = "cow",
        arguments = [posArg]
    )
    data = [
        ("cow goes woof", "cow woof woof"),
        ("cow goes moo", "cow moo"),
        ("cow goes meow", "cow meow meow meow")
    ] * 100
    random.shuffle(data)
    # Do training
    train_output = train.run_with_data_list(data, [cow], False, quiet_mode = True, num_epochs = 3)
    meta_model, final_state, train_iter, val_iter = train_output
    
    # eval the model. Expect should get basically perfect progam picking
    bashmetric = BashMetric()
    train.eval_model(meta_model, val_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98

def test_copy_mechanism():
    posArg = Argument("aposarg", "Stringlike", position = 0)
    cow = AIProgramDescription(
        name = "hello",
        arguments = [posArg]
    )
    traindata = [
        ("my name is bob", "hello bob"),
        ("my name is alice", "hello alice"),
        ("my name is Alice", "hello Alice"),
        ("my name is eve", "hello eve"),
        ("my name is jim", "hello jim"),
        ("my name is gregthefifth", "hello gregthefifth"),
    ]
    valdata = [
        ("my name is hoozawhatz", "hello hoozawhatz"),
        ("my name is boogieman", "hello boogieman"),
        ("my name is frankenstien", "hello frankenstien"),
        ("my name is walle", "hello walle"),
    ]

    # Do training
    train_output = train.run_with_specific_split(traindata, valdata, [cow], 
            False, quiet_mode = True, num_epochs = 200)
    meta_model, final_state, train_iter, val_iter = train_output
    
    # eval the model. Expect should get basically perfect progam picking
    bashmetric = BashMetric()
    train.eval_model(meta_model, train_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98, "Did not fully learn train"
    bashmetric = BashMetric()
    train.eval_model(meta_model, val_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98, "Did not generalize to val"

def test_copy_in_quotes():
    posArg = Argument("aposarg", "Stringlike", position = 0)
    cow = AIProgramDescription(
        name = "hello",
        arguments = [posArg]
    )
    traindata = [
        ('my name is "bob"', "hello bob"),
        ('my name is "alice"', "hello alice"),
        ('my name is "Alice"', "hello Alice"),
        ("my name is 'eve'", "hello eve"),
        ("my name is 'jim'", "hello jim"),
        ('my name is "gregthefifth"', "hello gregthefifth"),
    ]
    valdata = [
        ('my name is "hoozawhatz"', "hello hoozawhatz"),
        ('my name is "boogieman"', "hello boogieman"),
        ('my name is "frankenstien"', "hello frankenstien"),
        ("my name is 'walle'", "hello walle"),
    ]

    # Do training
    train_output = train.run_with_specific_split(traindata, valdata, [cow], 
            False, quiet_mode = True, num_epochs = 200)
    meta_model, final_state, train_iter, val_iter = train_output
    
    # eval the model. Expect should get basically perfect progam picking
    bashmetric = BashMetric()
    train.eval_model(meta_model, train_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98, "Did not fully learn train"
    bashmetric = BashMetric()
    train.eval_model(meta_model, val_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98, "Did not generalize to val"

# TODO (dngros): add version of the copy in quotes test to check whether it can
# actually see whether it is in quotes or not. So like it being in quotes effects
# the command chosen or something.
# TODO (dngros): test able to actually copy a true quote as well

def test_copy_long_seq():
    posArg = Argument("aposarg", "Stringlike", position = 0)
    cow = AIProgramDescription(
        name = "hello",
        arguments = [posArg]
    )
    traindata = [
        ("my name is a/b/c", "hello a/b/c"),
        ("my name is ab91.sb01", "hello ab91.sb01"),
        ("my name is John", "hello John"),
        ("my name is a/b/c/d/e/f/g", "hello a/b/c/d/e/f/g"),
        ("my name is a.b.cd.e.f.g", "hello a.b.cd.e.f.g"),
        ("my name is t.h.i.s.has.g.o.n.e.t.o.o.f.a.r", "hello t.h.i.s.has.g.o.n.e.t.o.o.f.a.r"),
    ]
    valdata = [
        ("my name is c/d/e/f/g/h/q", "hello c/d/e/f/g/h/q"),
        ("my name is how.long_can_this_be", "hello how.long_can_this_be"),
        ("my name is real.l.l.l.l.l.l.y", "hello  real.l.l.l.l.l.l.y"),
        ("my name is yo", "hello yo"),
    ]

    # Do training
    train_output = train.run_with_specific_split(traindata, valdata, [cow], 
            False, quiet_mode = True, num_epochs = 200)
    meta_model, final_state, train_iter, val_iter = train_output
    
    # eval the model. Expect should get basically perfect progam picking
    bashmetric = BashMetric()
    train.eval_model(meta_model, train_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98, "Did not fully learn train"
    bashmetric = BashMetric()
    train.eval_model(meta_model, val_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98, "Did not generalize to val"

def test_pipe_select():
    cow = AIProgramDescription(
        name = "cow"
    )
    dog = AIProgramDescription(
        name = "dog"
    )
    data = [
        ("I need a plumber", "cow | dog"),
        ("no plumber needed", "cow"),
        ("dog in my pipes", "dog | cow")
    ] * 100
    random.shuffle(data)
    # Do training
    train_output = train.run_with_data_list(data, [cow, dog], False, quiet_mode = True, num_epochs = 5)
    meta_model, final_state, train_iter, val_iter = train_output
    
    # eval the model. Expect should get basically perfect progam picking
    bashmetric = BashMetric()
    train.eval_model(meta_model, val_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98

def test_case_sensitive():
    posArg = Argument("aposarg", "Stringlike", position = 0)
    cow = AIProgramDescription(
        name = "hello",
        arguments = [posArg]
    )
    traindata = [
        ("my name is bob", "hello bob"),
        ("my name is Bob", "hello Bob"),
        ("my name is BoB", "hello BoB"),
        ("my name is boB", "hello boB"),
        ("my name is BOB", "hello BOB"),
    ]
    valdata = traindata

    # Do training
    train_output = train.run_with_specific_split(traindata, valdata, [cow], 
            False, quiet_mode = True, num_epochs = 200)
    meta_model, final_state, train_iter, val_iter = train_output
    
    # eval the model. Expect should get basically perfect progam picking
    bashmetric = BashMetric()
    train.eval_model(meta_model, train_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98, "Did not fully learn to gen case sensitive"

def test_fill_multival():
    aArg = Argument("a", "Stringlike")
    bArg = Argument("b", "Stringlike")
    cow = AIProgramDescription(
        name = "foo",
        arguments = [aArg, bArg]
    )
    data = [
        ("how are you", "foo -a good -b bad"),
        ("are you good or bad", "foo -a good -b bad"),
        ("flip please", "foo -a bad -b good")
    ] * 100
    random.shuffle(data)
    # Do training
    train_output = train.run_with_data_list(data, [cow], False, quiet_mode = True, num_epochs = 3)
    meta_model, final_state, train_iter, val_iter = train_output
    
    # eval the model. Expect should get basically perfect progam picking
    bashmetric = BashMetric()
    train.eval_model(meta_model, val_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98

def test_non_bow():
    """Test if can learn programs that based of queries that require something
    stronger than a bag-of-words assumption"""
    sad = AIProgramDescription(
        name = "sad",
    )
    what = AIProgramDescription(
        name = "what",
    )
    weird = AIProgramDescription(
        name = "weird",
    )
    traindata = [
        ("the dog bit bob", "sad"),
        ("bob bit the dog", "weird"),
        ("the bit dog bob", "what"),
        ("bob dog the bit", "what"),
    ]
    valdata = traindata

    # Do training
    train_output = train.run_with_specific_split(traindata, valdata, [sad, what, weird], 
            False, quiet_mode = True, num_epochs = 200)
    meta_model, final_state, train_iter, val_iter = train_output
    
    # eval the model. Expect should get basically perfect progam picking
    bashmetric = BashMetric()
    train.eval_model(meta_model, train_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98, "Did not fully learn non-bag-of-words"

def test_serialization():
    """Test a decently complex training task. Serialize the model. Then see if has same performance"""
    aArg = Argument("a", "Stringlike")
    bArg = Argument("b", "Stringlike")
    cow = AIProgramDescription(
        name = "cow",
        arguments = [aArg, bArg]
    )
    cArg = Argument("a", "Stringlike")
    dArg = Argument("b", "Stringlike")
    dog = AIProgramDescription(
        name = "dog",
        arguments = [cArg, dArg]
    )
    data = [
        ("have a bone puppy", "dog -a bone -b woof"),
        ("have a snack puppy", "dog -a snack -b woof"),
        ("have a apple puppy", "dog -a apple -b woof"),
        ("woof please", "dog -b woof"),
        ("nothin puppy", "dog"),
        ("nothin cow", "cow"),
        ("moo please", "cow -b moo"),
        ("have a grass cow", "cow -a grass -b moo"), ("have a plant cow", "cow -a plant -b moo"),
    ] * 100
    random.shuffle(data)
    # Do training
    train_output = train.run_with_data_list(data, [cow, dog], False, quiet_mode = True, num_epochs = 3)
    meta_model, final_state, train_iter, val_iter = train_output
    
    # eval the model. Expect should get basically perfect progam picking
    bashmetric = BashMetric()
    train.eval_model(meta_model, val_iter, [(bashmetric, 'bashmetric')])
    assert bashmetric.exact_match_acc() >= 0.98

    # Now serialize and restore
    fn = ".tester.pkl"
    serialize_tools.serialize(meta_model, fn)
    # delete stuff because suspicious of lingering state
    del meta_model, final_state, train_iter, bashmetric, aArg, bArg, cow, cArg, dArg, dog
    restored_model = serialize_tools.restore(fn)

    # check make sure has good preformance
    newmetric = BashMetric()
    train.eval_model(restored_model, val_iter, [(newmetric, 'bashmetric')])
    assert newmetric.exact_match_acc() >= 0.98, "Performance loss after restore"

# TODO make sure to actually get the filling in the serialization test
# TODO (DNGros): check the commands for vocab too
