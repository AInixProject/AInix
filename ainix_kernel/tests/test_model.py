import pytest
import pudb
from ainix_kernel.program_description import AIProgramDescription, Argument
from ainix_kernel.bashmetrics import BashMetric
import random
from model import SimpleCmd
import torchtext
from run_context import RunContext
import train as trainmodule

def test_arg_params_training():
    """A test to make sure that the arg parameters are being learned"""
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
    ] * 100
    (train, val), fields = trainmodule.build_dataset(data[:-10], data[-10:], [cow], False)
    (_, nl_field), (_, cmd_field) = fields 

    STD_WORD_SIZE = 8
    batch_size = 1
    context = RunContext(STD_WORD_SIZE, nl_field, cmd_field, [cow], False,
            batch_size = batch_size, debug = True, quiet_mode = True)

    train_iter = torchtext.data.iterator.BucketIterator(train,
        batch_size = batch_size, train = True, repeat = False,
        shuffle = True, sort_key=lambda x: x.nl,
        device = None if context.use_cuda else -1)
    val_iter = torchtext.data.iterator.BucketIterator(val,
        batch_size = batch_size, train = True, repeat = False,
        shuffle = True, sort_key=lambda x: x.nl,
        device = None if context.use_cuda else -1)

    meta_model = SimpleCmd(context)

    aArgData = aArg.model_data['top_v'].data.clone()
    bArgData = bArg.model_data['top_v'].data.clone()
    print("aArgData before:")
    print(aArgData)

    trainmodule.run_train(meta_model, train_iter, val_iter, context, num_epochs = 1)

    newAArgData = aArg.model_data['top_v'].data
    newBArgData = bArg.model_data['top_v'].data
    print("aArgData after:")
    print(newAArgData)
    print("aArgData before but after:")
    print(aArgData)

    assert not newAArgData.equal(aArgData)
    assert not newBArgData.equal(bArgData)

#def test_node_hidden():
#    cow = AIProgramDescription(
#        name = "cow"
#    )
#    dog = AIProgramDescription(
#        name = "dog"
#    )
#    data = [
#        ("I need a plumber", "cow | dog"),
#        ("no plumber needed", "cow"),
#        ("dog in my pipes", "dog | cow")
#    ] * 100
#    (train, val), fields = trainmodule.build_dataset(data[:-10], data[-10:], [cow], False)
#    (_, nl_field), (_, cmd_field) = fields 
#
#    STD_WORD_SIZE = 5
#    batch_size = 1
#    context = RunContext(STD_WORD_SIZE, nl_field, cmd_field, [cow], False,
#            batch_size = batch_size, debug = True, quiet_mode = True)
#
#    train_iter = torchtext.data.iterator.BucketIterator(train,
#        batch_size = batch_size, train = True, repeat = False,
#        shuffle = True, sort_key=lambda x: x.nl,
#        device = None if context.use_cuda else -1)
#    val_iter = torchtext.data.iterator.BucketIterator(val,
#        batch_size = batch_size, train = True, repeat = False,
#        shuffle = True, sort_key=lambda x: x.nl,
#        device = None if context.use_cuda else -1)
#
#    meta_model = SimpleCmd(context)
#
#    predNodeWeights = meta_model.predi.clone()
#    print("aArgData before:")
#    print(aArgData)
#
#    trainmodule.run_train(meta_model, train_iter, val_iter, context, num_epochs = 1)
#
#    newAArgData = aArg.model_data['top_v'].data
#    newBArgData = bArg.model_data['top_v'].data
#    print("aArgData after:")
#    print(newAArgData)
#    print("aArgData before but after:")
#    print(aArgData)
#
#    assert not newAArgData.equal(aArgData)
#    assert not newBArgData.equal(bArgData)
