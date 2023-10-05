import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

MEAN = [0.4913, 0.4821, 0.4465]
STD = [0.2023, 0.1994, 0.2010]

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def CIFAR10_data_transforms():
    MEAN = [0.4913, 0.4821, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(MEAN, STD)
        ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return train_transform, valid_transform

batch_size = 32
#batch_size = 16
#batch_size = 128
n_class = 10
learning_rate = 1e-3

momentum = 0.9
weight_decay = 5e-4

if __name__ == '__main__':

    data_dir = './'
    train_transform, valid_transform = CIFAR10_data_transforms()
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir, 'cifar10'), train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir, 'cifar10'), train=False, download=True, transform=valid_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    class AlexNet(nn.Module):

        def __init__(self, class_num):
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, class_num),
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            # print('in process x: ', x)
            x = self.classifier(x)
            return x
        
    class VGG(nn.Module):
        def __init__(self, class_num):
            super(VGG, self).__init__()
            # here is the part you need to finish
            self.features = nn.Sequential(
                # 32 x 32 x 3
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # 32 x 32 x 64
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 16 x 16 x 64
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # 16 x 16 x 128
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 8 x 8 x 128
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                # 8 x 8 x 256
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                # 8 x 8 x 256
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 4 x 4 x 256
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                # 4 x 4 x 512
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                # 4 x 4 x 512
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 2 x 2 x 512
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                # 2 x 2 x 512
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                # 2 x 2 x 512
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 1 x 1 x 512
            )
            self.classifier = nn.Sequential(
                nn.Linear(512 * 1 * 1, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, class_num),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            # here is the part you need to finish
            x = self.features(x)
            x = torch.flatten(x,1)
            x = self.classifier(x)
            return x

    class ResidualBlock(nn.Module):
        def __init__(self, inchannel, outchannel, stride:int=1, expansion:int=1, shortcut:nn.Module=None):
            super(ResidualBlock, self).__init__()
            self.expansion = expansion
            self.shortcut = shortcut
            self.features = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(outchannel),
            )

        def forward(self, x):
            x = self.features(x)
            self.shortcut = nn.Sequential()
            if (self.shortcut != None):
                out = self.shortcut(x)
            x = x + out
            x = nn.functional.relu(x)
            return x

    class ResNet(nn.Module):
        def __init__(self, class_num):
            super(ResNet, self).__init__()
            # here is the part you need to finish
            self.inchannel = 64
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                )
            self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
            self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
            self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
            self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
            self.fc = nn.Linear(512, class_num)

        def make_layer(self, block,  out_channel, num_blocks, stride):
            shortcut = None
            if (stride != 1):
                shortcut = nn.Sequential(
                    nn.Conv2d(self.inchannel, out_channel, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channel)
                )
            layers = []
            layers.append(block(self.inchannel, out_channel, stride, shortcut))
            self.inchannel = out_channel
            for i in range(0, num_blocks):
                layers.append(block(self.inchannel, out_channel))
            return nn.Sequential(*layers)

        def forward(self, x):
            # here is the part you need to finish
            x = self.features(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = nn.functional.avg_pool2d(x, 4)
            x = torch.flatten(x,1)
            x = self.fc(x)
            return x

    class Inception(nn.Module):
        def __init__(self, inchannel, outchannel1, inchannel3, outchannel3, inchannel5, outchannel5, finalchannel):
            super(Inception, self).__init__()
            # 1x1 conv branch
            self.branch1 = nn.Sequential(
                nn.Conv2d(inchannel, outchannel1, kernel_size=1),
                nn.BatchNorm2d(outchannel1),
                nn.ReLU(inplace=True),
            )
            # 1x1 conv -> 3x3 conv branch
            self.branch2 = nn.Sequential(
                nn.Conv2d(inchannel, inchannel3, kernel_size=1),
                nn.BatchNorm2d(inchannel3),
                nn.ReLU(inplace=True),
                nn.Conv2d(inchannel3, outchannel3, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel3),
                nn.ReLU(inplace=True),
            )
            # 1x1 conv -> 5x5 conv branch
            self.branch3 = nn.Sequential(
                nn.Conv2d(inchannel, inchannel5, kernel_size=1),
                nn.BatchNorm2d(inchannel5),
                nn.ReLU(inplace=True),
                nn.Conv2d(inchannel5, outchannel5, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel5),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel5, outchannel5, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel5),
                nn.ReLU(inplace=True),
            )
            # 3x3 pool -> 1x1 conv branch
            self.branch4 = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(inchannel, finalchannel, kernel_size=1),
                nn.BatchNorm2d(finalchannel),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            y1 = self.branch1(x)
            y2 = self.branch2(x)
            y3 = self.branch3(x)
            y4 = self.branch4(x)
            return torch.cat([y1,y2,y3,y4], 1)

    class GoogLeNet(nn.Module):
        def __init__(self, class_num):
            super(GoogLeNet, self).__init__()
            self.pre_layers = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
            )
            self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
            self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
            self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
            self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
            self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
            self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
            self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
            self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
            self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
            self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
            self.avgpool = nn.AvgPool2d(8, stride=1)
            self.linear = nn.Linear(1024, class_num)

        def forward(self, x):
            x = self.pre_layers(x)
            x = self.a3(x)
            x = self.b3(x)
            x = self.max_pool(x)
            x = self.a4(x)
            x = self.b4(x)
            x = self.c4(x)
            x = self.d4(x)
            x = self.e4(x)
            x = self.max_pool(x)
            x = self.a5(x)
            x = self.b5(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            return x
            
    model_name = 'alexnet'
    # [alexnet, vgg11, googlenet, resnet18]

    if model_name == 'alexnet':
        model = AlexNet(n_class).cuda()
    elif model_name == 'vgg11':
        model = VGG(n_class).cuda()
    elif model_name == 'resnet18':
        model = ResNet(n_class).cuda()
    elif model_name == 'googlenet':
        model = GoogLeNet(n_class).cuda()

    criterion = nn.CrossEntropyLoss()

    # SGD+momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # Nesterov Momentum
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    # RMSProp
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    # Adam
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    def train(epoch, model, train_loader, criterion, optimizer, scheduler):

        model.train()
        correct = 0
        start = time.time()

        for data, targets in train_loader:

            input, target = data.cuda(), targets.cuda()
            output = model(input)
            # print('output = ', output)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        scheduler.step() 
        accuracy = 100. * correct / len(train_loader.dataset)

        end = time.time()
        print(f'finish epoch {epoch+1} training in {end-start:.4f} s')

        return loss.item(), accuracy

    def test(model, test_loader):

        model.eval()
        correct = 0
        start = time.time()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
                test_loss = criterion(output, target)

            test_accuracy = 100. * correct / len(test_loader.dataset)

        end = time.time()
        print(f'finish epoch {epoch+1} testing in {end-start:.4f} s')

        return test_loss.item(), test_accuracy

    num_epoch = 50
    test_accuracy_list = []
    train_accuracy_list  = []
    train_loss_list = []
    test_loss_list = []
    for epoch in range(num_epoch):
        loss, train_accuracy = train(epoch, model, train_loader, criterion, optimizer, scheduler)
        test_loss, test_accuracy = test(model, test_loader)
        # ....
        #scheduler.step(test_accuracy)
        test_accuracy_list.append(test_accuracy)
        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(loss)
        test_loss_list.append(test_loss)
        print(f'finish epoch {epoch+1}, train loss: {loss:.4f}, train acc: {train_accuracy:.4f} %, test acc: {test_accuracy:.4f} %')

    import matplotlib.pyplot as plt
    # you need to finish
    plt.plot(train_accuracy_list)
    plt.plot(test_accuracy_list)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    #summarize history for loss
    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()