"""Attempts to unit test the various aspects the model
should be able to learn"""
import pytest
import pudb
import train
from program_description import AIProgramDescription, Argument
from bashmetrics import BashMetric
import random

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

def test_pipe_select():
    posArg = Argument("aposarg", "Stringlike", position = 0)
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
