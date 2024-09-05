import torch
from torch import nn
from torch import optim
from torchvision import models
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np


def get_model(model_name, device, pretrained=False, freeze = False):
    if model_name.startswith('vgg11'):
        model = VGG11(pretrained)
    
    if model_name.startswith('resnet18'):
        model = ResNet18(pretrained)
    
    if model_name.startswith('vgg16'):
        model = VGG16(pretrained, freeze)
        
    if model_name.startswith('vgg19'):
        model = VGG19(pretrained, freeze)
        
    elif model_name.startswith('resnet50'):
        model = ResNet50(pretrained)
        
    elif model_name.startswith('resnet152'):
        model = ResNet152(pretrained)
        
    elif model_name.startswith('inception'):
        model = Inception(pretrained)
    
    elif model_name.startswith('shallow_three_layer'):
        model = nn.Sequential(nn.Linear(8, 4), nn.Sigmoid(), nn.Linear(4, 2), nn.Sigmoid(), nn.Linear(2, 1)) 
        #model = Shallow_three_layer()
    
    
    elif model_name.startswith('OSA_shallow_three_layer'):
        model = nn.Sequential(nn.Linear(27, 10), nn.Sigmoid(), nn.Linear(10, 5), nn.Sigmoid(), nn.Linear(5, 1)) 
        
        
    elif model_name.startswith('saliva'):
        model = nn.Sequential(nn.Linear(59, 30), nn.Sigmoid(), nn.Linear(30, 15), nn.Sigmoid(), nn.Linear(15,5), nn.Sigmoid(), nn.Linear(5,1)) 
        
        
    return model.to(device)         

    
def get_last_conv_channel(model):
    layer_type_list = [nn.Conv2d, nn.ConvTranspose2d]
    return [module for module in model.base.modules() if type(module) in layer_type_list][-1].out_channels


class Inception(nn.Module):
    def __init__(self, pretrained, freeze):
        super(Inception, self).__init__()
        self.temp = models.inception_v3(pretrained=pretrained, aux_logits=True)
        self.temp.fc = nn.Linear(in_features=2048, out_features=1, bias=True).requires_grad_(not freeze)
    
    def forward(self, x):
        x = self.temp(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, pretrained):
        super(ResNet18, self).__init__()
        temp = models.resnet18(pretrained=pretrained)
        temp.fc = nn.Identity()        
        self.base = temp
        self.classifier = nn.Linear(in_features=get_last_conv_channel(self), out_features=1)
        
    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, pretrained, freeze):
        super(ResNet50, self).__init__()
        temp = models.resnet50(pretrained=pretrained)
        temp.fc = nn.Identity()        
        self.base = temp
        self.classifier = nn.Linear(in_features=get_last_conv_channel(self), out_features=1).requires_grad_(not freeze)
        
    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x


class ResNet152(nn.Module):
    def __init__(self, pretrained):
        super(ResNet152, self).__init__()
        self.base = models.resnet152(pretrained=pretrained)
        self.base.fc = nn.Linear(in_features=get_last_conv_channel(self), out_features=1)
        
    def forward(self, x):
        x = self.base_layer(x)
        return x

    
    
class VGG11(nn.Module):
    def __init__(self, pretrained):
        super(VGG11, self).__init__()
        self.base = models.vgg11_bn(pretrained=pretrained)
        self.base.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.base.classifier = nn.Linear(in_features=512, out_features=1, bias=True)

    def forward(self, x):
        x = self.base(x)
        return x      
    
class VGG16(nn.Module):
    def __init__(self, pretrained, freeze):
        super(VGG16, self).__init__()
        self.base = models.vgg16_bn(pretrained=pretrained)
        self.base.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.base.classifier = nn.Linear(in_features=512, out_features=1, bias=True).requires_grad_(not freeze)
        
    def forward(self, x):
        x = self.base(x)
        return x
        
class VGG19(nn.Module):
    def __init__(self, pretrained, freeze):
        super(VGG19, self).__init__()
        self.base = models.vgg19_bn(pretrained=pretrained)
        self.base.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.base.classifier = nn.Linear(in_features=512, out_features=1, bias=True).requires_grad_(not freeze)
        
    def forward(self, x):
        x = self.base(x)
        return x    
    
class Shallow_three_layer(nn.Module):
    def __init__(self):
        super(Shallow_three_layer, self).__init__()
        self.base = nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 2))
        self.base.classifier = nn.Linear(in_features=2, out_features=1, bias=True)
        
    def forward(self, x):
        x = self.base(x)
        return x


class ALEX(nn.Module):
    def __init__(self, pretrained):
        super(ALEX, self).__init__()
        self.base = models.alexnet.AlexNet(pretrained=pretrained)
        self.base.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.base.classifier = nn.Linear(in_features=256, out_features=1, bias=True)

    def forward(self, x):
        x = self.base(x)
        return x      

    
def train_convnet(model, 
                  device, 
                  train_loader, 
                  val_loader,
                  test_loader,
                  lr, 
                  epochs, 
                  save_path,
                  optimizer,
                  scheduler, 
                  min_epoch=2, 
                  pos_weight=1, 
                  save_by='val_auc', out = True): # val_auc
    
    # returning history
    history_train_loss = []
    history_val_loss = []
    history_val_auc = []
    history_test_auc = []
    history_train_auc = []
      
    # optimizer & learning rate
    model = model.to(device)   
    if pos_weight != 0:
        pos_weight = torch.as_tensor(pos_weight, dtype=torch.float)
        pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().cuda()
        
    train_auc = 0.0
    val_auc = 0.0
    test_auc = 0.0
    val_loss = 10000.0
    
    for epoch in range(1,epochs+1):
        running_loss = 0.0
        for data, label in train_loader:
            label = label.float()
            data = data.float()
            data, label = data.to(device), label.to(device)         
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        scheduler.step()
        
        ## training
        with torch.no_grad():
            probas = []
            train_ys = []
            for data, label in train_loader:
                data, label = data.to(device), label.to(device)
                data = data.float()
                label = label.float() # BCEloss requires float tensor
                proba = model(data)
                loss = criterion(proba, label)
                current_val_loss = loss.item()
                proba_np = proba.cpu().numpy()
                train_y = label.cpu().numpy()
                probas.append(proba_np)
                train_ys.append(train_y)
            probas = np.concatenate(probas)
            train_ys = np.concatenate(train_ys)
            train_auc = roc_auc_score(train_ys, probas)   
        
        # validation #
        current_val_loss = 0     
        current_test_loss = 0       
        model.eval()
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                data = data.float()
                label = label.float() # BCEloss requires float tensor
                proba = model(data)
                loss = criterion(proba, label)
                current_val_loss = loss.item()
                
                proba_np = proba.cpu().numpy()
                val_y = label.cpu().numpy()
                current_val_auc = roc_auc_score(val_y, proba_np)  
            
            # Test performance 
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                data = data.float()
                label = label.float()
                proba = model(data)
                loss = criterion(proba, label)
                current_test_loss = loss.item()
        
                proba_np = proba.cpu().numpy()
                test_y = label.cpu().numpy()
                current_test_auc = roc_auc_score(test_y, proba_np)   
        
        history_val_loss.append(current_val_loss)        
        history_val_auc.append(current_val_auc)
        history_train_loss.append(running_loss)
        history_test_auc.append(current_test_auc)
        history_train_auc.append(train_auc)
        
        # print log
        if out == True:
            print("epoch %d ended: train_auc = %.4f / val_loss = %.4f, val_auc = %.4f / test_loss = %.4f test_auc = %.4f" % (epoch, train_auc, current_val_loss, current_val_auc, current_test_loss, current_test_auc))  
                
        if epoch < min_epoch:
            model.train()
            continue
            
        # save best model    
        if save_by.startswith('val_loss'):
            if current_val_loss < val_loss:        
                torch.save(model.state_dict(), save_path)
                save_log = f"model validation loss down from {val_loss} to {current_val_loss}, save model to {save_path}"
                val_loss = current_val_loss
                #if out == True: print(save_log) 
        elif save_by.startswith('val_auc'):
            if current_val_auc > val_auc:                        
                torch.save(model.state_dict(), save_path)
                save_log = f"model validation auc improved from {val_auc} to {current_val_auc}, save model to {save_path}"
                val_auc = current_val_auc
                #if out == True: print(save_log)      
        elif save_by.startswith('test_auc'):
            if current_test_auc > test_auc:                        
                torch.save(model.state_dict(), save_path)
                save_log = f"model test auc improved from {test_auc} to {current_test_auc}, save model to {save_path}"
                test_auc = current_test_auc
                #if out == True: print(save_log)      
        # back to train
        model.train()
    
    # end of train
    return history_train_loss, history_val_loss, history_val_auc, history_test_auc, history_train_auc
