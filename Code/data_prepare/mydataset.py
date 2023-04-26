from PIL import Image
from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        imgs = []
        with open(txt_path, 'r') as fh:
            for line in fh:
                line = line.rstrip()
                words = line.split()  # list
                imgs.append((words[0], int(words[1])))

        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def collate_fn(batch):
        # 对批量样本做进一步的处理
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
