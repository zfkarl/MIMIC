import torchvision.datasets as dset
from torchvision import transforms
from torch.utils import data
import numpy as np

mean = (0.5412, 0.4323, 0.3796)
std = (0.2854, 0.2536, 0.2475)
        
normalize = transforms.Normalize(mean=mean, std=std)

train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.3),
    transforms.ToTensor(),
    normalize,
])

ms1m = dset.ImageFolder('/home/FaceData/ms1m/imgs',transform = train_transform ) 


if __name__ == '__main__':
     
    print(len(ms1m))
    #TrainLoader=data.DataLoader(train_set,batch_size=64,num_workers=8,shuffle=True)

    #print(TrainLoader.dataset.label.size())
    # for imgs,targets,idx in TrainLoader:
    #     print(imgs.shape)
    #     print(targets.shape)
    #     print(idx.shape)