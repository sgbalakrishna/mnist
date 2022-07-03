

#https://medium.com/swlh/pytorch-real-step-by-step-implementation-of-cnn-on-mnist-304b7140605a
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn



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