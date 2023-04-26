import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from data_prepare.mydataset import MyDataset
from model.resnet_model import resnet18, resnet34, resnet50, resnet101, resnet152
from model.vgg_model import vgg
from model.alex_model import AlexNet

# 设置随机数种子，确保结果可重复
torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device} device.")

# 训练集和验证集路径
train_txt_path = os.path.join("..", "train.txt")
valid_txt_path = os.path.join("..", "valid.txt")

# 数据预处理设置
normMean = [0.673, 0.639, 0.604]
normStd = [0.208, 0.209, 0.231]
normTransform = transforms.Normalize(normMean, normStd)

# 训练集的预处理
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normTransform
])

# 验证集的预处理
valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normTransform
])

batch_size = 32

# 构建MyDataset实例
train_dataset = MyDataset(txt_path=train_txt_path, transform=train_transform)
valid_dataset = MyDataset(txt_path=valid_txt_path, transform=valid_transform)

# 构建DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
validate_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

train_num = len(train_dataset)
val_num = len(valid_dataset)
print(f"using {train_num} images for training, {val_num} images for validation.")


# 构建模型
# net = AlexNet(num_classes=6, init_weights=True)

# pretrain
# As of v0.13, TorchVision offers a new Multi-weight support API
# for loading different weights to the existing model builder methods

# net = models.alexnet(pretrained=True)
# net = models.vgg19(pretrained=True)

net = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)  # new api

in_features = net.fc.in_features
net.fc = nn.Linear(in_features, 6)

# 初始化
nn.init.normal_(net.fc.weight, 0, 0.01)
nn.init.constant_(net.fc.bias, 0)


# dropout = 0.5
# 重写分类器
# net.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, 6),
#         )
#
# # 初始化权重
# for m in net.classifier.modules():
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, 0, 0.01)
#         nn.init.constant_(m.bias, 0)



# model_name = 'vgg11'
# net = vgg(model_name=model_name, num_classes=6, init_weights=True)

# net = resnet50(num_classes=6)

net.to(device)
loss_function = nn.CrossEntropyLoss()
learning_rate = 0.00003

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)  # 设置学习率下降策略
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

epochs = 240
# save_path = './AlexNet.pth'
# save_path = './save_weight/pretrain_vgg19.pth'
save_path = './save_weight/pretrain_resnet152.pth'
# save_path = f'{model_name}.pth'
# save_path = 'save_weight/resnet50.pth'


best_acc = 0.0
train_batch = len(train_loader)  # num of batches
val_batch = len(validate_loader)

for epoch in range(epochs):
    # train
    net.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)  # output state
    for data in train_bar:
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss.item())

    print(f'[epoch {epoch + 1}] 学习率:{scheduler.get_last_lr()}')
    scheduler.step()  # 更新学习率

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    val_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            loss = loss_function(outputs, val_labels.to(device))

            val_loss += loss.item()
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_bar.desc = "val epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss.item())

    val_accurate = acc / val_num

    print('[epoch %d] train_avg_loss: %.3f val_avg_loss: %.3f val_accuracy: %.3f' %
          (epoch + 1, train_loss / train_batch, val_loss / val_batch, val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

# torch.save(net.state_dict(), 'final.pth')
print(f'The Best acc is {best_acc:.3f}')
print('Finished Training')
