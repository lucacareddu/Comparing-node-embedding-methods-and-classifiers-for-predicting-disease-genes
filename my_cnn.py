import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn import BCELoss
from torch.optim import Adam

# import numpy as np
# from sklearn.metrics import accuracy_score

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(1, 32, 2, 1),
                                nn.ReLU()
                                )
                                
        self.max = nn.MaxPool1d(8, stride=7)

        self.fully = nn.Sequential(nn.Linear(576,288),
                                nn.ReLU(),
                                nn.Linear(288,144),
                                nn.ReLU(),
                                nn.Linear(144,1),
                                nn.Sigmoid()
                                )

    def forward(self, x):
        x = self.conv(x)
        x = self.max(torch.squeeze(x))
        return self.fully(torch.flatten(x,1))

class MY_CNN():
    def __init__(self, learning_rate=0.001, epochs=3000):
        self.cuda_on=torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_on else "cpu")
        
        self.dtype = torch.FloatTensor
        
        self.model = Net().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = BCELoss()

        self.epochs = epochs

    def predict(self, x_test):
        x_test=torch.from_numpy(x_test).type(self.dtype)[:, None, :, :]
        F.normalize(x_test)
        x_test = x_test.to(self.device)

        with torch.no_grad():
            self.model.eval()
            predictions = self.model(x_test)
            return torch.flatten(predictions, 1).cpu().numpy()
    
    def fit(self, x_train, y_train):
        x_train=torch.from_numpy(x_train).type(self.dtype)[:, None, :, :]
        F.normalize(x_train)
        y_train=torch.from_numpy(y_train).type(self.dtype)[:, None]

        x_train = Variable(x_train).to(self.device)
        y_train = y_train.to(self.device)

        print("\n\nMY_CNN running with CUDA: ", self.cuda_on)
        print("Number of epochs: ",self.epochs,"\n")

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            x_pred = self.model(x_train)

            loss = self.criterion(x_pred, y_train)

            loss.backward()
            self.optimizer.step()

            # print("Epoch : ",epoch+1, '\t', "loss :", loss.item())
            # if epoch%10==0:
            #     print("\nTest set prediction accuracy: ", accuracy_score(y_test, self.predict(x_test)), "\n")

# from torchsummary import summary
# model = Net().to("cuda")from torchsummary import summary
# model = Net().to("cuda")
# summary(model,input_size=(1,2,128))
# summary(model,input_size=(1,2,128))