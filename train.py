import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
from PIL import Image
import sys
from collections import OrderedDict


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    return trainloader, validloader, testloader, train_data

def define_network(learning_rate = 0.001, hidden_units = 500, arch = 'vgg16'):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13': 
        model = models.vgg13(pretrained=True)    
    # Freezing the pre-trained model parameters so we don't backprop gradients through them
    for param in model.parameters():
        param.requires_grad = False
    # Defining the OrderedDict for the new classifier portion of the network     
    if arch == 'vgg16':
        drop_p1 = 0.5
        drop_p2 = 0.5
        drop_p3 = 0.5
       
    elif arch == 'vgg13': 
        drop_p1 = 0.5
        drop_p2 = 0.5
        drop_p3 = 0.5

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(drop_p1)),
        ('fc2', nn.Linear(4096, 4096)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(drop_p2)),
        ('fc3', nn.Linear(4096, hidden_units)),
        ('relu3', nn.ReLU()),
        ('dropout3', nn.Dropout(drop_p3)),
        ('fc4', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier  
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    return model, criterion, optimizer

def check_accuracy_on_validation(validloader, model, device):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('Validation accuracy of the network: %d %%' % (100 * correct / total))

def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cuda'):
    epochs = epochs
    print_every = print_every
    steps = 0

    model.to(device)
    model.train()
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                model.eval()
                check_accuracy_on_validation(validloader, model, device)
                running_loss = 0
                model.train()

def check_accuracy_on_test(testloader, model, device):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('Test accuracy of the network: %d %%' % (100 * correct / total))

def saveCheckPoint(model, train_data, number_of_epochs, optimizer, criterion, number_of_hidden_units, filename = 'checkpoint.pth', save_directory = '', the_model_architecture_used = ''):
    model.class_to_idx = train_data.class_to_idx    
    checkpoint = {'class_to_idx': model.class_to_idx, 'epochs': number_of_epochs, 'classifier': model.classifier, 'arch': the_model_architecture_used, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'criterion': criterion.state_dict(), 'number_of_hidden_units': number_of_hidden_units} 

    if (len(save_directory.strip())>0):
        if not os.path.exists(save_directory.strip()):
            os.makedirs(save_directory)
        torch.save(checkpoint, save_directory+'/'+filename)
            
    else:
        torch.save(checkpoint, filename)

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_directory", type = str, default = 'flowers/', 
                        help = 'path to the folder of flower images')
    
    
    parser.add_argument('--save_directory', type = str, default = '', 
                        help = 'path to the folder of saved chackpoints')


    parser.add_argument('--arch', type = str, default = 'vgg16', 
                        help = 'CNN model architecture to be used: vgg13, vgg16') 
    

    parser.add_argument('--learning_rate', type = float, default = 0.001, 
                        help = 'learning rate of the network') 

    parser.add_argument('--epochs', type = int, default = 100, 
                        help = 'number of epochs of the network') 

    parser.add_argument('--hidden_units', type = int, default = 500, 
                        help = 'number of units in the last hidden layer') 

    parser.add_argument('--gpu', action = 'store_true', help = 'learning rate of the network')                      
   


    in_args = parser.parse_args()
    
    
    trainloader, validloader, testloader, train_data = load_data(in_args.data_directory)
    
    model, criterion, optimizer = define_network(in_args.learning_rate, in_args.hidden_units,                                                  in_args.arch)
    if in_args.gpu:
        if in_args.gpu and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'    
    else:
        device = 'cpu'
    
    print_every = 100
    
    do_deep_learning(model, trainloader, validloader, in_args.epochs, print_every, criterion,                      optimizer, device)
    
    model.eval()
    check_accuracy_on_test(testloader, model, device)
    
    saveCheckPoint(model, train_data, in_args.epochs, optimizer, criterion, in_args.hidden_units, 'checkpoint.pth', in_args.save_directory, in_args.arch, )
