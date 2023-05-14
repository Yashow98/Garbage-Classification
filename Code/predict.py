import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader

from data_prepare.mydataset import MyDataset
import model
from utils.confusion_matrix import ConfusionMatrix

# 设置随机数种子，确保结果可重复
torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device} device.")

test_txt_path = os.path.join("..", "test.txt")

# 数据预处理设置
normMean = [0.67254436, 0.639278, 0.6043134]
normStd = [0.208332, 0.2092541, 0.2310524]
normTransform = transforms.Normalize(normMean, normStd)

# 测试集的预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normTransform
])

batch_size = 32

test_dataset = MyDataset(test_txt_path, test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class_label = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
confusion = ConfusionMatrix(num_classes=6, class_labels=class_label)

# 构建模型
# net = model.AlexNet(num_classes=6)
# pretrain
# net = models.alexnet(num_classes=6)
# net = models.vgg19(num_classes=6)
net = models.resnet152(num_classes=6)

# model_name = 'vgg11'
# net = model.vgg(model_name=model_name, num_classes=6)

# net = model.resnet50(num_classes=6)

net.to(device)

# weights_path = "./AlexNet.pth"
# weights_path = f"./save_weight/{model_name}.pth"
# weights_path = 'save_weight/resnet50.pth'
# weights_path = 'save_weight/pretrain_vgg19.pth'
weights_path = 'save_weight/pretrain_resnet152.pth'

assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
net.load_state_dict(torch.load(weights_path))

net.eval()  # 去掉dropout，batch normalization使用全局均值和标准差
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = test_data
        outputs = net(test_images.to(device))
        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        confusion.update(outputs.cpu().numpy(), test_labels.cpu().numpy())
    confusion.summary()
    confusion.plot()
