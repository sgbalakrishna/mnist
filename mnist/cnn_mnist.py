

#https://medium.com/swlh/pytorch-real-step-by-step-implementation-of-cnn-on-mnist-304b7140605a
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

df_features = train.iloc[:,1:785]
df_label = train.iloc[:,0]



df_features.head()

X_train, X_cv, y_train, y_cv = train_test_split(df_features, df_label, 
                                                test_size = 0.2,
                                                random_state = 1212)

X_train = np.array(X_train).reshape(33600, 784) #(33600, 784)
X_cv = np.array(X_cv).reshape(8400, 784) #(8400, 784)


train_x = X_train.reshape(33600, 1,28,28)
train_x = torch.from_numpy(train_x).float()
y_train = torch.from_numpy(np.array(y_train))

y_train.shape

X_cv = X_cv.reshape(8400, 1, 28, 28)
X_cv  = torch.from_numpy(np.array(X_cv)).float()
# converting the target into torch format
y_cv = torch.from_numpy(np.array(y_cv))
X_cv.shape, y_cv.shape

# --- prepare batch ---

batch_size = 100

train = torch.utils.data.TensorDataset(train_x,y_train)
test = torch.utils.data.TensorDataset(X_cv,y_cv)

train_loader = torch.utils.data.DataLoader(train,batch_size = batch_size , shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)



#--- Model ---

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()

        #conv1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 5 * 5, 10) 

    def forward(self, x):
        # Set 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        # Set 2
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        
        #Flatten
        out = out.view(out.size(0), -1)

        #Dense
        out = self.fc1(out)
        
        return out#


#Definition of hyperparameters
n_iters = 2500
num_epochs = n_iters / (len(train_x) / batch_size)
num_epochs = int(num_epochs)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.001

model = CNNModel()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        train = Variable(images.view(100,1,28,28))
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        
        count += 1
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                
                test = Variable(images.view(100,1,28,28))
                # Forward propagation
                outputs = model(test)
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
