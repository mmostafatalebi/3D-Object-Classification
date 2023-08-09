#-----Libraries-------------------------------
import os
import glob
import trimesh
import numpy as np
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy
from torchmetrics import ConfusionMatrix
from torchvision.datasets.utils import download_and_extract_archive
import scipy
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time
import copy
import subprocess
#----------------------------------------------
#-----Download dataset-------------------------
url = "https://www.dropbox.com/s/ja56cvf3x4mkf1t/modelnet10_voxelized_32.npz"
output_filename = "modelnet10_voxelized_32.npz"
subprocess.run(["wget", url, "-O", output_filename])
filename = 'ModelNet10'
root = '~/tmp/'
download_and_extract_archive('http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip', root, filename)
DATA_DIR = 'ModelNet10/ModelNet10/'
#----------------------------------------------
#-----Data Visualization-----------------------
mesh = trimesh.load(os.path.join(DATA_DIR, "monitor/train/monitor_0377.off"))
mesh.show()
points = mesh.sample(1024)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
ax.set_axis_off()
plt.show()
vox = mesh.voxelized(5.0)
vox
vox = vox.revoxelized((32,32,32)).matrix.astype(int)
ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
ax.voxels(vox, edgecolor='k')
plt.show()
#----------------------------------------------
#-----Confusion Matrix-------------------------
def accurancy_per_class (conf_matrix, class_names):
    per_class_accuracy = 100 * torch.diag(conf_matrix) / torch.sum(conf_matrix, 1)
    tmp = {}
    for i, x in enumerate(class_names):
        tmp[x] = per_class_accuracy[i].item()
    print({"per class accuracy": tmp})
    
def create_matrix (conf_matrix, class_names ):
    fig = plt.figure(figsize = (12,7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', linewidths=.4, cbar=False)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names, rotation=0)
    plt.title("Confusion Matrix")
    accurancy_per_class(conf_matrix,class_names)
#----------------------------------------------
#-----Accurancy & Loss History-----------------    
def smooth(x, w=0.95):
    last = x[0]
    smoothed = []
    for point in x:
        smoothed_val = w * last + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def accurancy_and_loss_history (history ):
    eps = range(0, len(history["train_loss"].cpu()))
    sns.set_theme()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Results')
    ax[0].plot(eps, history["train_loss"].cpu(), 'g', label='Training Loss')
    ax[0].plot(eps, history["valid_loss"].cpu(), 'b', label='Valid Loss')
    ax[0].set_title('Loss History')
    ax[0].set(xlabel='Epochs', ylabel='Loss')
    ax[0].legend()

    ax[1].plot(eps, history["train_acc"].cpu(), 'g', label='Training Accuracy')
    ax[1].plot(eps, history["valid_acc"].cpu(), 'b', label='Valid Accuracy')
    ax[1].set_title('Accuracy History')
    ax[1].set(xlabel='Epochs', ylabel='Accuracy')
    ax[1].legend()
#----------------------------------------------
#-----Weight Histograms------------------------
def model_weights(model, t_cnn):
    # First hidden layer
    h1_w = model.conv3d_1.weight.data.cpu().numpy()
    h1_b = model.conv3d_1.bias.data.cpu().numpy()
    # Second hidden layer
    h2_w = model.conv3d_2.weight.data.cpu().numpy()
    h2_b = model.conv3d_2.bias.data.cpu().numpy()
    if t_cnn ==4:
      # Third hidden layer
      h3_w = model.conv3d_3.weight.data.cpu().numpy()
      h3_b = model.conv3d_3.bias.data.cpu().numpy()
      # Fouth hidden layer
      h4_w = model.conv3d_4.weight.data.cpu().numpy()
      h4_b = model.conv3d_4.bias.data.cpu().numpy()
    # Fifth hidden layer
    h5_w = model.fc1.weight.data.cpu().numpy()
    h5_b = model.fc1.bias.data.cpu().numpy()
    # Output layer
    out_w = model.fc2.weight.data.cpu().numpy()
    out_b = model.fc2.bias.data.cpu().numpy()
    
    if t_cnn ==4:
      fig, axs = plt.subplots(6, 1, figsize=(12,8))
      axs[0].hist(h1_w.flatten(), 50)
      axs[0].set_title('First hidden layer weights')
      axs[1].hist(h2_w.flatten(), 50)
      axs[1].set_title('Second hidden layer weights')
      axs[2].hist(h3_w.flatten(), 50)
      axs[2].set_title('Third layer weights')
      axs[3].hist(h4_w.flatten(), 50)
      axs[3].set_title('Fourth layer weights')
      axs[4].hist(h5_w.flatten(), 50)
      axs[4].set_title('Fifth layer weights')
      axs[5].hist(out_w.flatten(), 50)
      axs[5].set_title('Output layer weights')
    else:
      fig, axs = plt.subplots(4, 1, figsize=(12,8))
      axs[0].hist(h1_w.flatten(), 50)
      axs[0].set_title('First hidden layer weights')
      axs[1].hist(h2_w.flatten(), 50)
      axs[1].set_title('Second hidden layer weights')
      axs[2].hist(h5_w.flatten(), 50)
      axs[2].set_title('Fifth layer weights')
      axs[3].hist(out_w.flatten(), 50)
      axs[3].set_title('Output layer weights')
    [ax.grid() for ax in axs]
    plt.tight_layout()
    plt.show()
#----------------------------------------------
#-----Visualize Model--------------------------
def visualize_model(model, test_dataloader,class_names,num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig =  plt.figure(figsize=(30, 15))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                mesh =inputs.cpu().data[j]
                images_so_far += 1
                ax = fig.add_subplot(2,5,images_so_far,projection='3d',)
                ax.voxels(mesh[0], edgecolor='k')
                ax.axis('off')
                if class_names[labels[j]]!=class_names[preds[j]]:
                  ax.set_title(f'class: {class_names[labels[j]]} \n predicted: {class_names[preds[j]]}', loc="center",color='red')
                else:
                  ax.set_title(f'class: {class_names[labels[j]]} \n predicted: {class_names[preds[j]]}', loc="center",color='green')
                
                if images_so_far == num_images:
                    plt.show()
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
#----------------------------------------------
#-----Data Set---------------------------------        
class VoxelDataset(Dataset):
    
    def __init__(self,augment=True, train = True):
        self.augment=augment
        if train:
            tmp = np.load("modelnet10_voxelized_32.npz")
            self.data = tmp["X_train"]
            self.label = tmp["Y_train"]
            del tmp
        else:
            tmp = np.load("modelnet10_voxelized_32.npz")
            self.data = tmp["X_test"]
            self.label = tmp["Y_test"]
            del tmp
                
    def __len__(self):
      return len(self.label)

    def __preproc__(self, voxels): 
        #flip x
        if np.random.randint(2):
            voxels = np.flip(voxels, axis=0)
        #flip y
        if np.random.randint(2):
            voxels = np.flip(voxels, axis=1)
        return voxels.copy()

    def __getitem__(self, idx):
        label = self.label[idx]
        voxels = self.data[idx]
        if self.augment :
          voxels = self.__preproc__(voxels)
        voxels = np.expand_dims(voxels, axis=0)
        voxels = torch.tensor(voxels).float()
        return voxels, label
#----------------------------------------------
#-----Data Loader------------------------------    
train_ds = VoxelDataset(train=True)
test_ds = VoxelDataset(train=False)
train_ds.data[0].shape
train_dataloader = DataLoader(dataset=train_ds, batch_size=128, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_ds, batch_size=128)
mesh = next(iter(train_dataloader))
mesh = mesh[0][0][0]
ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
ax.voxels(mesh, edgecolor='k')
plt.show()
#----------------------------------------------
#-----Hyper Parameters-------------------------
learning_rate = 0.1
data_size = 32
sgd_momentum = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 60
lr_scheduler_step = 60
lr_scheduler_gamma = 0.5
print_epoch_rate = 1
verbose= 1
batch_size = 512
#----------------------------------------------
#-----Model Network----------------------------
class VoxelNet(nn.Module):
    def __init__(self,dropout,f1,f2,fc_1, n_classes=10, data_size=32):
        super().__init__()
        self.n_classes = n_classes
        self.data_size = data_size
        self.dropout=dropout
        self.f1 = f1
        self.f2 = f2
        self.fc_1 = fc_1
        
        #features
        self.conv3d_1 = nn.Conv3d(in_channels=1, out_channels=self.f1, kernel_size=5, stride=2)
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.conv3d_2 = nn.Conv3d(in_channels=self.f1, out_channels=self.f2, kernel_size=3)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.maxpool = nn.MaxPool3d(2)
         
        if data_size==64:
            dim=351232
            351232
        else:
           dim=27648
            
        #calculate dim after pooling for fc layer
        x = torch.rand((1,1,data_size,data_size,data_size))
        x= F.relu(self.conv3d_1(x))
        x = self.dropout1(x)
        x= F.relu(self.conv3d_2(x))
        x = self.dropout2(x)
        x = self.maxpool(x)
        print(x.shape)
        dim = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]
        #mlp
        self.fc1 = nn.Linear(dim, fc_1)
        self.dropout3 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(self.fc_1, self.n_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        #features
        x=F.relu(self.conv3d_1(x))
        x = self.dropout1(x)
        x=F.relu(self.conv3d_2(x))
        x = self.dropout2(x)
        #maxpool
        x = self.maxpool(x)
        #flatten
        x = x.view(x.size(0), -1)
        #mlp
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return self.logsoftmax(x)

model = VoxelNet(dropout = 0.2,f1= 64,f2 =64,fc_1=64, data_size=32)    
class_names = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
num_classes = len(class_names)
#----------------------------------------------
#-----Optimizer & Loss Fucntion----------------
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)
loss_func = nn.NLLLoss()
#----------------------------------------------
#-----Training---------------------------------
train_loss_history, valid_loss_history = [], []
train_acc_history, valid_acc_history = [], []
train_accuracy = Accuracy(task='multiclass', num_classes=10)
valid_accuracy = Accuracy(task='multiclass', num_classes=10)
print("Started Training...\n")
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
total_time = time.time()
for epoch in range(0, epochs):            
    t = time.time()
    train_loss = []                                                         #track training loss
    valid_loss = []                                                         #track valid loss
    #track loss for 10 batch
    batch_loss=0
    #training on batches
    model.train()
    for i, data in enumerate(train_dataloader, 0):
        x, y = data[0].to(device), data[1].to(device)                        #send to device
        optimizer.zero_grad()
        pred = model(x)                                                      #predict class
        loss = loss_func(pred, y)                                            #compute and track loss
        train_loss.append(loss.cpu().data)                                   #track loss
        acc = train_accuracy(torch.argmax(pred, 1).cpu(), y.cpu())           #track accuracy
        #back propagate and optimize
        loss.backward()
        optimizer.step()
        if lr_scheduler != None:
          lr_scheduler.step()
    
    pred_conf = []                                                           #track for confusion matrix
    y_conf = []                                                              #track for confusion matrix
    #validation on batches
    model.eval()
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)                                       #send to device
        pred = model(x)                                                         #predict class
        loss = loss_func(pred, y)                                               #compute and track loss
        valid_loss.append(loss.cpu().data)                                      #track loss
        valid_accuracy.update(torch.argmax(pred, 1).cpu(), y.cpu())             #track accuracy
        pred_conf.append(torch.argmax(pred, 1))
        y_conf.append(y)
    # total accuracy over all batches
    total_train_accuracy = train_accuracy.compute()
    total_valid_accuracy = valid_accuracy.compute()
    train_accuracy.reset()
    valid_accuracy.reset()
    #track loss and acc for plotting
    train_loss_history.append(torch.mean(torch.tensor(train_loss)))
    valid_loss_history.append(torch.mean(torch.tensor(valid_loss)))
    train_acc_history.append(total_train_accuracy)
    valid_acc_history.append(total_valid_accuracy)
    if total_valid_accuracy > best_acc:
        best_acc = total_valid_accuracy
        best_model_wts = copy.deepcopy(model.state_dict())
        #compute confusion matrix
        a = torch.cat(pred_conf).cpu()
        b = torch.cat(y_conf).cpu()
        confmat = ConfusionMatrix(task='multiclass', num_classes=10)
        conf_matrix = confmat(a, b)

    elapsed_time_epoch = time.time() - t                                   #compute epoch time
    tmp1 = "epoch:{:3d}/{:3d}".format(epoch+1, epochs)
    tmp2 = "train-loss: {:4.2f}, train-acc: {:.2%}".format(train_loss_history[epoch], train_acc_history[epoch].item())
    tmp3 = "valid-loss: {:4.2f}, valid-acc: {:.2%}".format(valid_loss_history[epoch], valid_acc_history[epoch].item())
    tmp4 = "time: {:.2f} seconds".format(elapsed_time_epoch)
    print(tmp1, tmp2, tmp3, tmp4)
    print({"train loss": train_loss_history[epoch], "epoch": epoch})
    print({"valid loss": valid_loss_history[epoch], "epoch": epoch})
    print({"train accuracy": train_acc_history[epoch].item(), "epoch": epoch})
    print({"valid accuracy": valid_acc_history[epoch].item(), "epoch": epoch})
   
#save history
history = {"train_loss": torch.tensor(train_loss_history), "train_acc": torch.tensor(train_acc_history), \
          "valid_loss": torch.tensor(valid_loss_history), "valid_acc": torch.tensor(valid_acc_history)}
elapsed_time_training = time.time() - total_time   
#print end results
print("Finished Training\n")
idc = torch.argmax(torch.tensor(valid_acc_history)).item()
print("time: {:.2f} seconds".format(elapsed_time_training))
print("time: {:.2f} minutes".format(elapsed_time_training/60))
print("Avg Time Per Epoch {:.2f} seconds".format(elapsed_time_training/epochs))
print("Best Val Acc: {:.2%}".format(best_acc))
# load best model weights
model.load_state_dict(best_model_wts) 
#----------------------------------------------
#-----Results of Confusion Matrix--------------
create_matrix(conf_matrix,class_names)
#----------------------------------------------
#-----History of Loss and Accuracy-------------
accurancy_and_loss_history(history)
#----------------------------------------------
#-----Weight Histograms------------------------
model_weights(model,2)
#----------------------------------------------
#-----Visualize Model--------------------------
visualize_model(model, test_dataloader, class_names)
#----------------------------------------------
#-----Classification Report--------------------
print(classification_report(b,a, target_names =class_names))
#----------------------------------------------
#-----Orion Implemetntaion and Load Data-------
def load_data(batch_size, augment):
    train_ds = VoxelDataset(augment= augment, train=True)
    test_ds = VoxelDataset(augment=augment,train=False)
    train_ds.data[0].shape
    train_dataloader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_ds, batch_size=batch_size)
    return train_dataloader,test_dataloader

train_dataloader, test_dataloader = load_data(512, True)
mesh = next(iter(train_dataloader))
mesh = mesh[0][0][0]
ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
ax.voxels(mesh, edgecolor='k')
plt.show()
#----------------------------------------------
#-----Model Network----------------------------
class Orion(nn.Module):
    def __init__(self,f1,f2,f3,f4,fc_1,p, n_classes=10, data_size=32):
        super().__init__()
        self.n_classes = n_classes
        self.data_size = data_size
        self.f1 = f1
        self.f2 = f2
        self.f3=  f3
        self.f4=  f4
        self.fc_1 = fc_1
        self.p = p  # dropout ratios
        #features
        self.conv3d_1 = nn.Conv3d(in_channels=1, out_channels=self.f1, kernel_size=(3,3,3), stride=2,padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(self.f1)
        self.dropout1 = nn.Dropout(p=self.p[0])
        self.avgpool1 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3d_2 = nn.Conv3d(in_channels=self.f1, out_channels=self.f2, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(self.f2)
        self.dropout2 = nn.Dropout(p=self.p[1])
        self.avgpool2 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
       
        self.conv3d_3 = nn.Conv3d(in_channels=f2, out_channels=self.f3, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))       
        self.bn3 = nn.BatchNorm3d(self.f3)
        self.dropout3 = nn.Dropout(p=self.p[2])
        self.avgpool3 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3d_4 = nn.Conv3d(in_channels=self.f3, out_channels=self.f4, kernel_size=(3,3,3), stride=1,padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(self.f4)
        self.dropout4 = nn.Dropout(p=self.p[3])        
        self.avgpool4 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
         
        if data_size==64:
            dim=351232
            351232
        else:
           dim=27648
            
        #calculate dim after pooling for fc layer
        x = torch.rand((1,1,data_size,data_size,data_size))
        x =self.conv3d_1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.avgpool1(x)

        x=self.conv3d_2(x)
        x = self.bn2(x)
        x= F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.avgpool2(x)

        x = self.conv3d_3(x)
        x = self.bn3(x)
        x= F.leaky_relu(x)
        x = self.dropout3(x)
        x = self.avgpool3(x)

        x = self.conv3d_4(x)
        x = self.bn4(x)
        x= F.leaky_relu(x)
        x = self.dropout4(x)
        x = self.avgpool4(x)
        
        dim = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]
        #mlp
        self.fc1 = nn.Linear(dim, fc_1)
        self.dropout5 = nn.Dropout(p=self.p[4])
        self.fc2 = nn.Linear(self.fc_1, self.n_classes)
        #self.logsoftmax = nn.LogSoftmax(dim=1)
        
        
    def forward(self, x):
        #features
        x = self.bn1(self.conv3d_1(x))
        x= F.rrelu(x)
        x = self.dropout1(x)
        x = self.avgpool1(x)

        x = self.bn2(self.conv3d_2(x))
        x= F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.avgpool2(x)

        x = self.bn3(self.conv3d_3(x))
        x= F.leaky_relu(x)
        x = self.dropout3(x)
        x = self.avgpool3(x)

        x = self.bn4(self.conv3d_4(x))
        x= F.leaky_relu(x)
        x = self.dropout4(x)
        x = self.avgpool4(x)
        
        #flatten
        x = x.view(x.size(0), -1)
        #mlp
        x= F.leaky_relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)
        return x
#----------------------------------------------
#-----Trainig----------------------------------
class_names = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
num_classes = len(class_names)
def train (model, epochs,loss_func, optimizer,  train_dataloader, test_dataloader,lr_scheduler):
  train_loss_history, valid_loss_history = [], []
  train_acc_history, valid_acc_history = [], []
  train_accuracy = Accuracy(task='multiclass', num_classes=10)
  valid_accuracy = Accuracy(task='multiclass', num_classes=10)
  print("started training...\n")
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  total_time = time.time()
  for epoch in range(0, epochs):            
      t = time.time()
      train_loss = []                                                         #track training loss
      valid_loss = []                                                         #track valid loss
      #track loss for 10 batch
      batch_loss=0
      #training on batches
      model.train()
      for i, data in enumerate(train_dataloader, 0):
          x, y = data[0].to(device), data[1].to(device)                        #send to device
          optimizer.zero_grad()
          pred = model(x)                                                      #predict class
          loss = loss_func(pred, y)                                            #compute and track loss
          train_loss.append(loss.cpu().data)                                   #track loss
          acc = train_accuracy(torch.argmax(pred, 1).cpu(), y.cpu())           #track accuracy
          #back propagate and optimize
          loss.backward()
          optimizer.step()
          lr_scheduler.step()
      pred_conf = []                                                           #track for confusion matrix
      y_conf = []                                                              #track for confusion matrix
      #validation on batches
      model.eval()
      for x, y in test_dataloader:
          x, y = x.to(device), y.to(device)                                       #send to device
          pred = model(x)                                                         #predict class
          loss = loss_func(pred, y)                                               #compute and track loss
          valid_loss.append(loss.cpu().data)                                      #track loss
          valid_accuracy.update(torch.argmax(pred, 1).cpu(), y.cpu())             #track accuracy
          pred_conf.append(torch.argmax(pred, 1))
          y_conf.append(y)
      # total accuracy over all batches
      total_train_accuracy = train_accuracy.compute()
      total_valid_accuracy = valid_accuracy.compute()
      train_accuracy.reset()
      valid_accuracy.reset()
      #track loss and acc for plotting
      train_loss_history.append(torch.mean(torch.tensor(train_loss)))
      valid_loss_history.append(torch.mean(torch.tensor(valid_loss)))
      train_acc_history.append(total_train_accuracy)
      valid_acc_history.append(total_valid_accuracy)
      if total_valid_accuracy > best_acc:      
        best_acc = total_valid_accuracy
        best_model_wts = copy.deepcopy(model.state_dict())
        #compute confusion matrix
        a = torch.cat(pred_conf).cpu()
        b = torch.cat(y_conf).cpu()
        confmat = ConfusionMatrix(task='multiclass', num_classes=10)
        conf_matrix = confmat(a, b)
        
      elapsed_time_epoch = time.time() - t   
      tmp1 = "epoch:{:3d}/{:3d}".format(epoch+1, epochs)
      tmp2 = "train-loss: {:4.2f}, train-acc: {:.2%}".format(train_loss_history[epoch], train_acc_history[epoch].item())
      tmp3 = "valid-loss: {:4.2f}, valid-acc: {:.2%}".format(valid_loss_history[epoch], valid_acc_history[epoch].item())
      tmp4 = "time: {:.2f} seconds".format(elapsed_time_epoch)
      print(tmp1, tmp2, tmp3, tmp4)
      print({"train loss": train_loss_history[epoch], "epoch": epoch+1})
      print({"valid loss": valid_loss_history[epoch], "epoch": epoch+1})
      print({"train accuracy": train_acc_history[epoch].item(), "epoch": epoch+1})
      print({"valid accuracy": valid_acc_history[epoch].item(), "epoch": epoch+1})
  #save history
  history = {"train_loss": torch.tensor(train_loss_history), "train_acc": torch.tensor(train_acc_history), \
            "valid_loss": torch.tensor(valid_loss_history), "valid_acc": torch.tensor(valid_acc_history)}
  elapsed_time_training = time.time() - total_time   
  #print end results
  print("finished training\n")
  idc = torch.argmax(torch.tensor(valid_acc_history)).item()
  print("Time: {:.2f} seconds".format(elapsed_time_training))
  print("Time: {:.2f} minutes".format(elapsed_time_training/60))
  print("Avg Time Per Epoch {:.2f} seconds".format(elapsed_time_training/epochs))
  print("Best Val Acc: {:.2%}".format(best_acc))
  # load best model weights
  model.load_state_dict(best_model_wts)
  return conf_matrix, history,model,b, a      
#-----Experinces for Batch Size and Learning Rate-----
batch_size=512
learning_rate = 0.1
data_size = 32
sgd_momentum = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 60
lr_scheduler_step = 60
lr_scheduler_gamma = 0.5
print_epoch_rate = 1
p = [0.2, 0.3, 0.4, 0.6, 0.4] # dropout ratios
f= [32,64, 128, 256] # features
fc_1 = 128
n = Orion(f1=f[0],f2=f[1], f3=f[2], f4=f[3], fc_1=fc_1, p=p, n_classes=10, data_size=32)
n.get_parameter
#-----------------------------------------------------
#-----Optimizer & Loss Function-----------------------
optimizer = torch.optim.SGD(n.parameters(), lr=learning_rate, momentum=sgd_momentum)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)
loss_func = nn.CrossEntropyLoss()
#-----------------------------------------------------
#-----Train Data Loader-------------------------------
train_dataloader, test_dataloader = load_data(batch_size=batch_size, augment=True)
conf_matrix, history,model,y_conf, pred = train(model=n, epochs=epochs,loss_func=loss_func, \
                             train_dataloader=train_dataloader, test_dataloader=test_dataloader, \
                             optimizer=optimizer, lr_scheduler =lr_scheduler)
#-----------------------------------------------------
#-----Results of Confusion Matrix---------------------
create_matrix(conf_matrix,class_names)
#-----------------------------------------------------
#-----History of Loss and Accuracy--------------------
accurancy_and_loss_history(history)
#-----------------------------------------------------
#-----Weight Histograms-------------------------------
model_weights(model,4)
#-----------------------------------------------------
#-----Visualize Model---------------------------------
visualize_model(model, test_dataloader, class_names)
#-----------------------------------------------------
#-----Classification Report---------------------------
print(classification_report(y_conf, pred, target_names =class_names))
#-----------------------------------------------------