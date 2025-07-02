import torch
from RINN import rinn, reparameterize_model
import torch.optim
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time

from torch.utils.data import DataLoader
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_data = torchvision.datasets.CIFAR10(root="C:/Cifar10", train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize(256),torchvision.transforms.RandomResizedCrop(224,scale=(0.5,1.0))]),
                                         download=True)
train_data = torchvision.datasets.CIFAR10(root="C:/Cifar10", train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize(256),torchvision.transforms.RandomResizedCrop(224,scale=(0.5,1.0))]),
                                         download=True)
# train_data = torchvision.datasets.ImageFolder(root="C:/ImageNet/train", transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize(256),torchvision.transforms.RandomResizedCrop(224,scale=(0.5,1.0))]))
# test_data = torchvision.datasets.ImageFolder(root="C:/ImageNet/val", transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize(256),torchvision.transforms.RandomResizedCrop(224,scale=(0.5,1.0))]))
# train_data=train_data[0]
# train_data=train_data[0:5000]
# test_data=test_data[0]
# test_data=test_data[0:1000]

# length
train_data_size = len(train_data)
test_data_size = len(test_data)

# DataLoader
train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64)
# model
model = rinn(num_classes=100, variant='s0')
model = model.to(device)

loss_fu = nn.CrossEntropyLoss()
loss_fu = loss_fu.to(device)
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epoch=300
writer = SummaryWriter("logs_network")
# start_time = time.time()
for j in range(epoch):
    print("-----------{}epoch-----------".format(j + 1))
    model.train()
    total_train_loss = 0.0
    total = 0
    correct = 0
    start_time = time.time()
    for i, data in enumerate(train_data_loader,0): # 1 step = 1 batch = 8    1 step = 1 batch = 64
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device) # torch.Size([8])
        output = model(imgs)  # (8, 100)
        loss = loss_fu(output, targets) # 8 batch. tensor(4.6368, device='cuda:0', grad_fn=<NllLossBackward>)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        if (i+1) % 100 == 0:
            # end_time = time.time()
            print("step:{},loss:{}, accuracy:{}".format(i+1, loss.item(), correct/total*100))
            filename = "./R0/Train_sampled_loss_per_100steps.txt"
            with open(filename, 'a') as file_object:
                file_object.write("{}\n".format(loss.item()))
            writer.add_scalar("train_loss", loss.item(), i+1)
    end_time = time.time()
    filename = "./R0/Train_GPUtime_per_epoch.txt"
    with open(filename, 'a') as file_object:
        file_object.write("{}\n".format(end_time - start_time))
    filename = "./R0/Train_average_loss_per_epoch.txt"
    with open(filename, 'a') as file_object:
        file_object.write("{}\n".format(total_train_loss/len(train_data_loader)))
    filename = "./R0/Train_average_accuracy_per_epoch.txt"
    with open(filename, 'a') as file_object:
        file_object.write("{}\n".format(correct/total*100))

    model.eval()
    model_eval = reparameterize_model(model)
    model_eval = model_eval.to(device)
    total_test_loss = 0
    total_test_accuracy = 0
    total=0
    correct=0
    start_time = time.time()
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model_eval(imgs)
            loss = loss_fu(outputs, targets)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs.data,1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        end_time = time.time()
        filename = "./R0/Test_GPUtime_per_epoch.txt"
        with open(filename, 'a') as file_object:
            file_object.write("{}\n".format(end_time - start_time))
        filename = "./R0/Test_average_loss_per_epoch.txt"
        with open(filename, 'a') as file_object:
            file_object.write("{}\n".format(total_test_loss / len(test_data_loader)))
        filename = "./R0/Test_average_accuracy_per_epoch.txt"
        with open(filename, 'a') as file_object:
            file_object.write("{}\n".format(correct / total * 100))

        print("{}step sumloss:{}".format(j + 1, total_test_loss))
        print("{}step loss:{}".format(j + 1, total_test_loss / len(test_data_loader)))
        print("{}step accuracy:{}".format(j + 1, correct / total * 100))
        # torch.save(model.state_dict(), "./R0/modelk=3_3_no_dconv_train(batch=64)/model_eval_{}.pth".format(j))
        # torch.save(model_eval.state_dict(),"./R0/modelk=3_3_no_dconv_test(batch=64)/model_eval_{}.pth".format(j))
        print("model saved")
writer.close()

