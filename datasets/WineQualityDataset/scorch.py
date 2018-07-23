import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing
from skorch.net import NeuralNetClassifier
from skorch.net import NeuralNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0
TRAIN_SIZE = 400 - TEST_SIZE

# Load Data from CSV File
data = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)
# scaler = preprocessing.StandardScaler().fit(X_train)
np.random.shuffle(data)
X = data[:TRAIN_SIZE, 0:-1]
y = torch.from_numpy(data[:TRAIN_SIZE:, -1] - 1).long()


# Define Network Architecture
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(21, 17)
        self.layer2 = torch.nn.Linear(17, 13)
        self.layer3 = torch.nn.Linear(13, 9)
        self.layer4 = torch.nn.Linear(9, 5)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        out1 = F.relu(self.layer1(x))
        out1 = self.dropout(out1)
        out2 = F.relu(self.layer2(out1))
        out3 = F.relu(self.layer3(out2))
        y_pred = F.softmax(self.layer4(out3))

        return y_pred.float()


net = NeuralNetClassifier(
    Model,
    max_epochs=1000,
    lr=0.01
    # use_cuda=True,  # uncomment this to train with CUDA
)

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('net', net),
])

pipe.fit(X, y)