import pudb
import pickle

class Interface():
    def __init__(self):
        STD_WORD_SIZE = 80
        self.use_cuda = False
        with open(r"../ainix_kernel/savedtest.pkl", "rb") as output_file:
            self.model = pickle.load(output_file)    

    def predict(self, utterance):
        pass
