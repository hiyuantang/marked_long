import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

print('Just wait. It takes a while!')
data_path = '/home/AD/yutang/markedlong_dataset'
traindir = os.path.join(data_path, 'train')
batch_size = 64

train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(160),
            transforms.ToTensor(),
        ]))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers = 4, pin_memory=True)

def get_mean_sd(loader):
    mean = 0.
    sd = 0.
    total_image_num = 0
    for images, _ in loader:
        count = images.size()[0]
        images = images.view(count, images.size()[1], -1)
        
        total_image_num = total_image_num+count
        mean = mean+images.mean(2).sum(0)
        sd = sd+images.std(2).sum(0)
    mean = mean/total_image_num
    sd = sd/total_image_num
    return mean, sd

mean, sd = get_mean_sd(train_loader)
print(mean, sd)