import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt

#from plots import plot9images, plot_confusion_matrix, plot_histogram_from_labels
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("PyTorch version = {} ".format(torch.__version__))

print('CUDA available? %s' % torch.cuda.is_available())

train_X = np.load('train_data.npy') / 255
train_y = np.load('train_labels.npy')
valid1_X = np.load('valid_data1.npy') / 255
valid1_y = np.load('valid_labels1.npy')
valid2_X = np.load('valid_data2.npy') / 255
valid2_y = np.load('valid_labels2.npy')
test_X = np.load('test_track_data.npy') / 255

valid1_X = valid1_X.reshape((-1, 3, 45, 80))
valid2_X = valid2_X.reshape((-1, 3, 45, 80))
test_X = test_X.reshape((-1, 3, 45, 80))
train_X = train_X.reshape((-1, 3, 45, 80))

def labels2csv(labels, csv_path):
    with open(csv_path, "w") as file:
        file.write("id,label\n")
        for i, label in enumerate(labels):
            file.write("{},{}\n".format(i,label))


# In[3]:


# define hyperparams

class LRConfig(object):
    """
    Holds logistic regression model hyperparams.
    
    :param height: image height
    :type heights: int
    :param width: image width
    :type width: int
    :param channels: image channels
    :type channels: int
    :param batch_size: batch size for training
    :type batch_size: int
    :param epochs: number of epochs
    :type epochs: int
    :param save_step: when step % save_step == 0, the model
                      parameters are saved.
    :type save_step: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float
    """
    def __init__(self,
                 height=45,
                 width=80,
                 channels=3,
                 classes=3,
                 architecture=[100, 3],
                 conv_architecture=[12, 16],
                 kernel_sizes=None,
                 pool_kernel=None,
                 save_step=100,
                 batch_size=32,
                 epochs=1,
                 learning_rate=0.0054,
                 momentum=0.1,
                 weight_decay=0.1):
        if kernel_sizes is None:
            self.kernel_sizes = [5] * len(conv_architecture)
        else:
            self.kernel_sizes = kernel_sizes
        if pool_kernel is None:
            self.pool_kernel = [2] * len(conv_architecture)
        else:
            pool_kernel = self.pool_kernel
        self.height = height
        self.width = width
        self.channels = channels
        self.classes = classes
        self.architecture = architecture
        self.conv_architecture = conv_architecture
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_step = save_step
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        

    def __str__(self):
        """
        Get all attributs values.
        :return: all hyperparams as a string
        :rtype: str
        """
        status = "height = {}\n".format(self.height)
        status += "width = {}\n".format(self.width)
        status += "channels = {}\n".format(self.channels)
        status += "classes = {}\n".format(self.classes)
        status += "architecture = {}\n".format(self.architecture)
        status += "conv_architecture = {}\n".format(self.conv_architecture)
        status += "kernel_sizes = {}\n".format(self.kernel_sizes)
        status += "pool_kernel = {}\n".format(self.pool_kernel)
        status += "batch_size = {}\n".format(self.batch_size)
        status += "epochs = {}\n".format(self.epochs)
        status += "learning_rate = {}\n".format(self.learning_rate)
        status += "momentum = {}\n".format(self.momentum)
        status += "save_step = {}\n".format(self.save_step)
        status += "weight_decay = {}\n".format(self.weight_decay)

        return status


# In[4]:


# set hyperparams

lr_config = LRConfig()
lr_config.epochs = 15
lr_config.learning_rate = 0.01
lr_config.momentum = 0.1
lr_config.architecture = [500, 100, 3]
lr_config.weight_decay = 0.0001
print("Os hiper parametros do modelo de regressao logistica sao:\n")
print(lr_config)


# In[5]:


# organizing data

class DataHolder():
    """
    Class to store all data.

    :param config: hyper params configuration
    :type config: LRConfig or DFNConfig
    :param train_dataset: dataset of training data
    :type train_dataset: torch.utils.data.dataset.TensorDataset
    :param test_dataset: dataset of test data
    :type test_dataset: torch.utils.data.dataset.TensorDataset
    :param valid_dataset: dataset of valid data
    :type valid_dataset: torch.utils.data.dataset.TensorDataset
    :param batch_size: batch size for training
    :type test_batch: batch size for the testing data
    :param test_batch: int
    """
    def __init__(self,
                 config,
                 train_dataset,
                 valid_dataset,
                 test_dataset,
                 real_data):
        batch_size = config.batch_size        
        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.valid_loader = DataLoader(dataset=valid_dataset,
                                       batch_size=len(valid_dataset))
        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=len(test_dataset))
        self.real_data = torch.Tensor(real_data)        

train_dataset = TensorDataset(torch.Tensor(train_X),
                              torch.Tensor(train_y).type(torch.LongTensor))
valid1_dataset = TensorDataset(torch.Tensor(valid1_X),
                              torch.Tensor(valid1_y).type(torch.LongTensor))
valid2_dataset = TensorDataset(torch.Tensor(valid2_X),
                              torch.Tensor(valid2_y).type(torch.LongTensor))
self_driving_data = DataHolder(lr_config, train_dataset, valid1_dataset, valid2_dataset, test_X) 


# In[6]:


# model trainer

def train_model_img_classification(model,
                                   config,
                                   dataholder,
                                   criterion, 
                                   optimizer):

    train_loader = dataholder.train_loader
    valid_loader = dataholder.valid_loader

    train_loss = []
    valid_loss = []
    for epoch in range(config.epochs):
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels) 
                    
            loss.backward()
            optimizer.step()
            
            if step % config.save_step == 0:
                train_loss.append(float(loss))
            
                # Test on validation data
                (v_images, v_labels) = next(iter(valid_loader))
                v_pred = model(v_images)
                v_loss = criterion(v_pred, v_labels)
                valid_loss.append(float(v_loss))

        v_candidates = model.predict(v_images)
        acc = v_candidates.eq(v_labels).sum()
        print("End of epoch %d | Validation loss: %.3f | Accuracy: %d / %d" % (epoch, float(v_loss), acc, len(v_labels)))
        acc = v_candidates.eq(v_labels).sum()
                        
    # Plot
 #   x = np.arange(1, len(train_loss) + 1, 1)
 #   _, ax = plt.subplots(1, 1, figsize=(12, 5))
 #   ax.plot(x, train_loss, label='train loss')
 #   ax.plot(x, valid_loss, label='valid loss')
 #   ax.legend()
 #   plt.xlabel('step')
 #   plt.ylabel('loss')
 #   plt.title('Train and valid loss')
 #   plt.grid(True)
 #   plt.show()
        


# In[7]:

class CNN(nn.Module):
    def __init__(self,
                 config):
        super(CNN, self).__init__()        
        self.num_conv = len(config.kernel_sizes)
        self.num_linear = len(config.architecture)
                
        # adding convolution layers
        in_chans = config.channels
        for step, (kernel, out_chans, pool_kernel) in enumerate(zip(config.kernel_sizes, config.conv_architecture, config.pool_kernel)):
            self.add_module("conv"+str(step), nn.Conv2d(in_chans, out_chans, kernel))
            self.add_module("pool"+str(step), nn.MaxPool2d(pool_kernel))
            in_chans = out_chans
            
        # adding fully connected layers

        p_in = self.calc_fc_size(config)
        for step, p_out in enumerate(config.architecture):
            self.add_module("lin"+str(step), nn.Linear(p_in, p_out))
            p_in = p_out        
                
    # calculate FC layer dimension based on convolutional layer configuration
    def calc_fc_size(self, config):
        chans = 3
        height = 45
        width = 80
        
        # calculate outcome from convolutional layers and pooling layers
        for (kernel, out_chans, pool_kernel) in zip(config.kernel_sizes, config.conv_architecture, config.pool_kernel):
            chans = out_chans
            height -= kernel-1
            width -= kernel-1
            height = int(height / pool_kernel)
            width = int(width / pool_kernel)
        
        return chans * height * width
        

    def forward(self, x):
        inn = x
        for i in range(self.num_conv):
            conv = getattr(self, "conv"+str(i))
            inn = conv(inn)
            inn = F.relu(inn)
            pool = getattr(self, "pool"+str(i))
            inn = pool(inn)
            
        inn = inn.view(inn.shape[0], -1)
        for i in range(self.num_linear):
            linear = getattr(self, "lin"+str(i))
            inn = linear(inn)

        return inn
    
    
    def predict(self, x):
        logits = self.forward(x)
        probs = nn.functional.softmax(logits, 1)
        predictions = probs.argmax(dim=1)        
        return predictions 


# In[ ]:


# define tools and train

model = CNN(lr_config)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr_config.learning_rate, lr_config.momentum, weight_decay=lr_config.weight_decay)

train_model_img_classification(model,
                               lr_config,
                               self_driving_data,
                               criterion,
                               optimizer)


# In[9]:


# run over test data and plot confusion

img, labels = next(iter(self_driving_data.test_loader))
pred = model.predict(img).numpy()

#plot_confusion_matrix(truth=labels.numpy(),
#                      predictions=pred,
#                      save=False,
#                      path="dfn_confusion_matrix.png",
#                      classes=["forward", "left", "right"])


# In[275]:


# run over submittable data and save CSV
preds = model.predict(self_driving_data.real_data).numpy()
labels2csv(preds, "submission.csv")

