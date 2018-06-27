import pickle

default_file_name = "savetest.pkl"
def serialize(meta_model, filename = default_file_name):
    with open(filename, "wb") as output_file:
        pickle.dump(meta_model, output_file)    

def restore(filename = default_file_name):
    with open(filename, "rb") as output_file:
        model = pickle.load(output_file)    
    return model
