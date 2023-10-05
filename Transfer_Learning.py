import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_train_valid_loader(
    data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True, amount_of_data=1
):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # the amount of training data is decided here
    num_train = round(len(train_dataset) * amount_of_data)
    print("Amount of training data: ", num_train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return (train_loader, valid_loader)

def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader

if __name__ == '__main__':
    # CIFAR10 dataset
    # the parameter "amount_of_data" decide the amount of training data
    train_loader, valid_loader = get_train_valid_loader(
        data_dir="./data", batch_size=64, augment=False, random_seed=1, amount_of_data=0.1
    )

    test_loader = get_test_loader(data_dir="./data", batch_size=64)

    # Alexnet arichtecture
    class AlexNet(nn.Module):
        def __init__(self, num_classes = 1000, dropout = 0.5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # +++++++++++++++++++++++++++++++++++++
    imagenet_num_classes = 1000
    cifar_num_classes = 10
    # +++++++++++++++++++++++++++++++++++++

    # parameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.005

    # Select the model
    model_name = 'alexnet_tran'
    # [alexnet, alexnet_tran, vgg16_tran, resnet18_tran]

    # Select True if you want to change the layers
    # Therefore, change to True for question 3C
    modify_whole_classifier = False

    from torchvision.datasets.utils import download_url
    import torchvision.models as models

    if model_name == 'alexnet':
        model = AlexNet(cifar_num_classes).to(device)
    elif model_name == 'alexnet_tran':
        model = AlexNet(imagenet_num_classes).to(device)

        download_url('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth', '.\content')
        state_dict = torch.load("./content/alexnet-owt-7be5be79.pth")
        model.load_state_dict(state_dict)

        if modify_whole_classifier:
            model.classifier = nn.Sequential()
            model.classifier.add_module("0", nn.Dropout(p=0.5))
            model.classifier.add_module("1", nn.Linear(256 * 6 * 6, 4096))
            model.classifier.add_module("2", nn.ReLU(inplace=True))
            model.classifier.add_module("3", nn.Dropout(p=0.5))
            model.classifier.add_module("4", nn.Linear(4096, 4096))
            model.classifier.add_module("5", nn.ReLU(inplace=True))
            model.classifier.add_module("6", nn.Linear(4096, cifar_num_classes))
            model.to(device)
        else:
            model.classifier[6] = nn.Linear(4096, cifar_num_classes)
            model.to(device)

    elif model_name == 'vgg16_tran':
        # Use the VGG16 model in pytorch library
        # Set the weights to None just for better file management
        model = models.vgg16(weights=None).to(device)
        print(model)

        # Download the weight of pre-trained VGG16 and save the file to the current folder
        download_url('https://download.pytorch.org/models/vgg16-397923af.pth', '.\content')
        state_dict = torch.load("./content/vgg16-397923af.pth")
        model.load_state_dict(state_dict)
 
        if modify_whole_classifier:
            model.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
            model.classifier = nn.Sequential()
            model.classifier.add_module("0", nn.Linear(512 * 3 * 3, 512))
            model.classifier.add_module("1", nn.ReLU(inplace=True))
            model.classifier.add_module("2", nn.Dropout(p=0.5))
            model.classifier.add_module("3", nn.Linear(512, 128))
            model.classifier.add_module("4", nn.ReLU(inplace=True))
            model.classifier.add_module("5", nn.Dropout(p=0.5))
            model.classifier.add_module("6", nn.Linear(128, cifar_num_classes))
            model.to(device)
            print(model)
        else:
            model.classifier[6] = nn.Linear(4096, cifar_num_classes)
            model.to(device)

    elif model_name == 'resnet18_tran':
        # Use the resnet18 model in pytorch library
        # Set the weights to None just for better file management
        model = models.resnet18(weights=None).to(device)
        
        # Download the weight of pre-trained resnet18 and save the file to the current folder
        download_url('https://download.pytorch.org/models/resnet18-5c106cde.pth', '.\content')
        state_dict = torch.load("./content/resnet18-5c106cde.pth")
        model.load_state_dict(state_dict)

        model.fc = nn.Linear(512, cifar_num_classes)
        model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

    total_step = len(train_loader)

    print('{} - Training start'.format(str(datetime.now())))

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('{} - Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(str(datetime.now()), epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))
