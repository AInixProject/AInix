# ...
class Model(nn.Module):
    def __init__(self, num_of_classes):
        super(Model, self).__init__()
        # ...
        self.predict_if_valid_fc = nn.Linear(hidden_size, num_of_classes)
        self.predict_preference_fc = nn.Linear(hidden_size, num_of_classes)
    
    def forward(self, x):
        # ... hidden layer operations on x
        is_valid_prediction = self.predict_if_valid_fc(x)
        log_softmax_preference_prediction = F.log_softmax(self.predict_preference_fc(x), dim = 1)
        return is_valid_prediction, log_softmax_preference_prediction
# ...
model = Model(num_of_classes)
# ...

def get_loss(x, y):
    """y is mini-batched tensor of length num_of_classes representing ground-truth labeling.
    For example y = [[0,3,1,0,0]] would represent that y[0][1] and y[0][2] are valid labelings, 
    but that y[0][2] is three times as preferable as y[0][1]"""
    is_valid_prediction, log_softmaxes = model(x)
    y_logits = (y > 0).type(torch.FloatTensor)
    class_loss = F.binary_cross_entropy_with_logits(is_valid_prediction, y_logits)
    preference_loss = F.kl_div(log_softmaxes, F.normalize(y))
    return class_loss + preference_loss

# optimize model on get_loss....

def get_prediction(x):
    """try to output which class is most probable, and the probability that class is valid"""
    is_valid_prediction, log_softmaxes = model(x)
    _, most_likely_class = log_softmaxes.max(1)
    probability_valid = F.sigmoid(is_valid_prediction[most_likely_class]
    return most_likely_class, probability_valid)
