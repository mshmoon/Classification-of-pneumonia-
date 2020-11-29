import os
import numpy as np
import time
import sys


import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import f1_score
from DensenetModels import DenseNet121

from DatasetGenerator import DatasetGenerator_train
from DatasetGenerator import DatasetGenerator_test

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def load_densenet(model, model_path):
    kwargs = {'map_location': lambda storage, loc: storage.cuda(0)}
    state_dict = torch.load(model_path, **kwargs)
    state_dict=state_dict['state_dict']
    a_list= model.state_dict().keys()
    j=0
    for k in a_list:
        name1 = k
        name2 = "module."+k
        try:
            name2=name2.replace("conv","conv.")
            name2=name2.replace("norm","norm.")
            model.state_dict()[name1][:]=state_dict[name2][:]
        except:
            j=j+1
    return model

class ChexnetTrainer():

    def train (pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):

        which_model=["resnet34","resnet50","resnet101","resnest","densenet"]
        model_name=which_model[-1]

        model=ChexnetTrainer.define_model(model_name,nnClassCount, nnIsTrained)
        if model_name == "densenet":
            model.densenet121.classifier=nn.Linear(1024,2)
        else:
            if model_name=="resnet34":
                model.fc=nn.Linear(512,2)
            if model_name=="resnet50" or model_name=="resnet101":
                model.fc=nn.Linear(2048,2)

        model=model.cuda()
       # model.load_state_dict(torch.load('0.8564356435643564_model.pth'), strict=True)
        dataLoaderTrain, dataLoaderTest = ChexnetTrainer.preprocess(model_name)
        loss = torch.nn.BCELoss(size_average=True)
        base_lr = 0.01
        total_step=(558//8+1)*60
        cur_step=0
        for epochID in range(60):
            model.train() if epochID<=30 else model.eval()
            epoch1=[]
            step=0
            correct_num=0
            sample_num=0
            f1_score_pred = []
            target_set = []
            auc_score_set = []
            for batchID, (input, target) in enumerate (dataLoaderTrain):
                learning_rate = base_lr * (total_step - cur_step) / total_step
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
                cur_step+=1
                step+=1
                input=input.cuda()
                target = target.cuda()
                if step%5==3:
                    time.sleep(3)
                # -------------------only for densenet121------------------
                # out,pde_loss= model(input.requires_grad_(True),flag="pde")
                # ---------------------------------------------------------
                out= model(input)
                pred = torch.argmax(out,1)
                label = torch.argmax(target,1)
                for i in range(len(pred)):
                    sample_num+=1
                    if pred[i] == label[i]:
                        correct_num += 1
                accuracy = correct_num /sample_num
                crossentropy = loss(torch.sigmoid(out), target)
                epoch1.append(crossentropy)
                avg_entropy=sum(epoch1)/(batchID+1)
                total_loss=crossentropy
                for i in range(input.size()[0]):
                    f1_score_pred.append(pred[i].cpu().data.numpy().item())
                    target_set.append(label[i].cpu().data.numpy().item())
                    auc_score_set.append(out[i].squeeze()[label[i]].cpu().data.numpy().item())

                f1_s = ChexnetTrainer.f1_score_func(f1_score_pred, target_set)
                auc_s = ChexnetTrainer.auc_func(target_set, auc_score_set)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                print(f"epoch {epochID} step {step} average_entrpy {avg_entropy} accuracy {accuracy} f1_s {f1_s} auc_s {auc_s}")
            time.sleep(30)
            with torch.no_grad():
                ChexnetTrainer.epochVal(model, dataLoaderTest,loss)

    def epochVal(model, dataLoader,loss):
        model.eval ()
        correct_num=0
        f1_score_pred=[]
        target_set=[]
        auc_score_set = []
        epoch1 = []
        for i, (input, target) in enumerate (dataLoader):
            target = target.cuda()
            input = input.cuda()
            out= model(input)
            crossentropy = loss(torch.sigmoid(out), target)
            epoch1.append(crossentropy)
            pred = torch.argmax(out)
            label = torch.argmax(target)

            f1_score_pred.append(pred.cpu().data.numpy().item())
            target_set.append(label.cpu().data.numpy().item())
            auc_score_set.append(out.squeeze()[label].cpu().data.numpy().item())
            if pred==label:
                correct_num+=1
        accuracy=correct_num/i
        f1_s=ChexnetTrainer.f1_score_func(f1_score_pred,target_set)
        auc_s=ChexnetTrainer.auc_func(target_set,auc_score_set)
        avg_entropy = sum(epoch1) / (i + 1)
        print(f"--eval-- avg_entropy {avg_entropy} accuracy {accuracy} f1_cores {f1_s} auc_scores {auc_s}")
        torch.save(model.state_dict(), f"{accuracy}_model.pth")

    def auc_func(target,score):
        tar1,sco1,tar2,sco2=[],[],[],[]
        for i in range(len(target)):
            if target[i]==0:
                tar1.append(target[i])
                sco1.append(score[i])
            else:
                tar2.append(target[i])
                sco2.append(score[i])
        del target
        del score
        tar=tar1+tar2
        sco=sco1+sco2
        tar=np.array(tar)
        sco=np.array(sco)
        return roc_auc_score(tar,sco)

    def f1_score_func(pred,target):
        return f1_score(target,pred)

    def define_model(model_name,nnClassCount, nnIsTrained):
        if model_name=="resnet34":
            model = torchvision.models.resnet34(pretrained=True)
            return model
        if model_name=="resnet50":
            model = torchvision.models.resnet50(pretrained=True)
            return model
        if model_name=="resnet101":
            model = torchvision.models.resnet101(pretrained=True)
            return model
        if model_name=="resnest":
            torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
            model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
            return model
        if model_name=="densenet":
            pathModel = "models/m-25012018-123527.pth.tar"
            model = DenseNet121(nnClassCount, nnIsTrained)
            model = load_densenet(model, pathModel)
            return model

    def preprocess(model_name):
        transformList1 = []
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList1.append(transforms.Resize((224, 224)))
        transformList1.append(transforms.RandomHorizontalFlip())
        transformList1.append(transforms.ToTensor())
        transformList1.append(normalize)
        transformSequence1 = transforms.Compose(transformList1)

        transformList2 = []
        transformList2.append(transforms.Resize((224, 224)))
        transformList2.append(transforms.ToTensor())
        transformList2.append(normalize)
        transformSequence2 = transforms.Compose(transformList2)

        datasetTrain = DatasetGenerator_train(pathImageDirectory="D:\someprogram\dataset",
                                              transform=transformSequence1,model_name=model_name)
        datasetTest = DatasetGenerator_test(pathImageDirectory="D:\someprogram\dataset",
                                            transform=transformSequence2,model_name=model_name)
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        return dataLoaderTrain,dataLoaderTest





