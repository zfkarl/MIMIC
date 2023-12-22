import torchvision.datasets as dset
from torchvision import transforms
from torch.utils import data
import numpy as np

mean = (0.5,0.5,0.5)
std = (0.5, 0.5,0.5)
        
normalize = transforms.Normalize(mean=mean, std=std)

train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.3),
    transforms.ToTensor(),
    normalize,
])

glint = dset.ImageFolder('/home/zhangfan/dataset/glint360k/output',transform = train_transform ) 

# label = []
# for i in range(len(dataset1)):
#     label.append(dataset1[i][1])

# class Glint360k(data.Dataset):
#     def __init__(self):

#         self.dataset1 = dataset1
        
#         self.label = np.array(dataset1)[:,1]

#     def __getitem__(self,index):
#         img = self.dataset1[index][0]
#         label = self.dataset1[index][1]
        
#         return img, label, index

#     def __len__(self):
#         return len(self.dataset1)

if __name__ == '__main__':
    
    #print(len(Glint360k.classes))   

    train_set = glint()
    print(len(train_set))
    TrainLoader=data.DataLoader(train_set,batch_size=64,num_workers=8,shuffle=True)

    #print(TrainLoader.dataset.label.size())
    # for imgs,targets,idx in TrainLoader:
    #     print(imgs.shape)
    #     print(targets.shape)
    #     print(idx.shape)