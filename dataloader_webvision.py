from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os

class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir+'imagenet/val/'
        self.transform = transform
        self.val_data = []
        for c in range(num_class):
            imgs = os.listdir(self.root+str(c))
            for img in imgs:
                self.val_data.append([c,os.path.join(self.root,str(c),img)])                
                
    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')   
        img = self.transform(image) 
        return img, target
    
    def __len__(self):
        return len(self.val_data)

class webvision_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode, num_class, pred=[], probability=[], log=''): 
        self.root = root_dir
        self.transform = transform
        self.mode = mode  
     
        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target                             
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    train_imgs.append(img)
                    self.train_labels[img]=target            
            if self.mode == 'all':
                self.train_imgs = train_imgs
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                
                    self.probability = [probability[i] for i in pred_idx]            
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))            
                    log.write('Numer of labeled samples:%d \n'%(pred.sum()))
                    log.flush()                          
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                           
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))            
                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, index        
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    


class webvision_dataloader():  
    def __init__(self, batch_size, num_class, num_workers, root_dir, log):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log

        self.transform_train = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])  
        self.transform_imagenet = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])         

    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="all", num_class=self.num_class)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)                 
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="labeled",num_class=self.num_class,pred=pred,probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)        
            
            unlabeled_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",num_class=self.num_class,pred=pred,log=self.log)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test', num_class=self.num_class)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='all', num_class=self.num_class)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return eval_loader     
        
        elif mode=='imagenet':
            imagenet_val = imagenet_dataset(root_dir=self.root_dir, transform=self.transform_imagenet, num_class=self.num_class)      
            imagenet_loader = DataLoader(
                dataset=imagenet_val, 
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return imagenet_loader     

