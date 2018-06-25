import ainix_kernel.train
from ainix_kernel.run_context import RunContext
# required to get to descs. TODO move
import ainix_kernel.data as sampledata
class Interface():
    def __init__():
        STD_WORD_SIZE = 80
        self.use_cuda = False
        (train, val), fields = train.build_dataset([], [], sampledata.all_desc, self.use_cuda)
        (_, nl_field), (_, cmd_field) = fields 
        self.context = train.RunContext(STD_WORD_SIZE, nl_field, cmd_field, descs, 
                self.use_cuda, batch_size = 1, debug = True, quiet_mode = False)
        self.model = SimpleCmd(self.context)

    def predict(utterance):
        pass
