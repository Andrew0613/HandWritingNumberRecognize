from matplotlib.transforms import Transform
from numpy import float32, save
from sympy import true
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import *
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import os 
import time
from torchstat import stat
from torchsummary import summary
from tqdm import tqdm
from torchvision import models
total_iters = 0
class HandWritingNumberRecognize_Dataset(Dataset):
    def __init__(self,opt,phase):
        """
        input:
        opt: hyperparameters
        phase: experient phase
        
        output:
        
        """
        # 这里添加数据集的初始化内容
        self.dataroot = opt.dataroot
        self.dir = os.path.join(self.dataroot,phase)
        self.img_dir = os.path.join(self.dir,"images")
        self.label_dir = None
        if phase == "train":
            self.label_dir = os.path.join(self.dir,"labels_train.txt")
        elif phase == "val":
            self.label_dir = os.path.join(self.dir,"labels_val.txt")
        self.paths = make_dataset(self.img_dir, opt.max_dataset_size)
        self.paths.sort(key=lambda x : int(x.split('.')[0].split('_')[1] ))
        # paths = sorted(glob.glob(self.img_dir+"/*.bmp"))
        self.labels = []
        if self.label_dir:
            with open(self.label_dir,"r") as f :
                for line in f.readlines():
                    line = line.strip("\n")
                    self.labels.append(line)
        self.size  = len(self.paths)
        self.transform = self.get_transform()
    def get_transform(self):
        """
        input:None
        output: transforms 
        """
        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        return transforms.Compose(transform_list)
    def __getitem__(self, index):
        """
        input:
        index: index of input image
        output: 
        dic: a dictionary of image and its label or a dictionary of image
        """
        path = self.paths[index % self.size]  # make sure index is within then range
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        if len(self.labels)>0:
            label = int(self.labels[index % self.size])
            return {'img': img, 'label': label}
        # apply image transformation
        return {'img': img}
    def __len__(self):
        # 这里添加len函数的相关内容
        """
        input: None
        output: length of Dataset
        """
        return self.size

#l3_w_512
class HandWritingNumberRecognize_Network(torch.nn.Module):
    def __init__(self,opt):
        """
        input:hyperparameters
        output: a network
        """
        super(HandWritingNumberRecognize_Network, self).__init__()
        self.gpu_ids = opt.gpu_ids
        self.model_name  = 'FC'
        # 此处添加网络的相关结构，下面的pass不必保留
        self.layer1 = nn.Sequential(#构造一个sequential容器
            nn.Linear(in_features=784*3, out_features=256),
            # nn.Linear(in_features=784*3, out_features=2048).to(device),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(256),  # N, H, W
            # nn.LayerNorm(512),  # C, H, W
            # nn.InstanceNorm1d(512),  # H, W  (要求输入数据三维)
            # nn.GroupNorm(2, 512)  # C, H, W,  将512分成两组
            nn.ReLU()
        )  # N, 512
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            # nn.Linear(in_features=2048, out_features=2048).to(device),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )  # N, 256
        # self.layer3 = nn.Sequential(
        #     # nn.Linear(in_features=256, out_features=10).to(device)
        #     nn.Linear(in_features=512, out_features=512).to(device),
        #     nn.Dropout(0.25),
        #     nn.BatchNorm1d(512),
        #     # nn.ReLU()
        # )  # N, 128
        # self.layer4 = nn.Sequential(
        #     # nn.Linear(in_features=256, out_features=10).to(device)
        #     nn.Linear(in_features=512, out_features=512).to(device),
        #     nn.Dropout(0.5),
        #     nn.BatchNorm1d(512),
        #     # nn.ReLU()
        # )  # N, 128
        self.layer3 = nn.Sequential(
             nn.Linear(in_features=256, out_features=10)
            # nn.Linear(in_features=2048, out_features=10).to(device),
        )  # N, 10
    def forward(self, input_data):
        """
        input: data
        output: prediction
        """
        # 此处添加模型前馈函数的内容，return函数需自行修改
        x = self.layer1(input_data)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        return x


def validation(data_loader_val):
    """
    input:data_loader for validation
    output: accuracy
    """
    # 验证函数，任务是在训练经过一定的轮数之后，对验证集中的数据进行预测并与真实结果进行比对，生成当前模型在验证集上的准确率
    correct = 0
    total = 0
    accuracy = 0
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for data in data_loader_val:
            images, true_labels = data['img'],data['label']
            # 在这一部分撰写验证的内容，下面两行不必保留
            images = images.view(-1,3*28*28).to(device)
            labels = true_labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0)  ##更新测试图片的数量   size(0),返回行数
            correct += (predicted == labels).sum().item() ##更新正确分类的图片的数量
    accuracy = correct/total
    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", accuracy)
    return accuracy

def alltest(data_loader_test):
    """
    input:data_loader for test
    output: a txt file of prediction
    """
    # 测试函数，需要完成的任务有：根据测试数据集中的数据，逐个对其进行预测，生成预测值。
    # 将结果按顺序写入txt文件中，下面一行不必保留
    res_dir = os.path.join(opt.results,opt.name)
    model =  HandWritingNumberRecognize_Network(opt).to(device)
    model_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model_path = os.path.join(model_dir, 'latest_net_FC.pth')
    model.load_state_dict(torch.load(model_path)) #load model
    results = []
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for data in data_loader_test:
            images = data['img']
            images = images.view(-1,3*28*28).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) 
            results.append(predicted)
    
    save_results(results,res_dir,"prediction",is_prediction=True)
        
def train(epoch_num,data_loader_train,opt):
    """
    input:
    epoch_num: which epoch it is now
    data_loader_train: a dataloader for training
    opt: hyperparameters

    output: training time for this epoch 
    """
    # 循环外可以自行添加必要内容
    global total_iters
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    epoch_start_time = time.time()  # timer for entire epoch
    epoch_iter = 0 
    total_step = len(data_loader_train)
    for index, data in enumerate(data_loader_train, 0):
        iter_data_time = time.time()    # timer for data loading per iteration
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        
        images, true_labels = data['img'],data['label']
        true_labels = true_labels.to(device)
        images = images.view(-1,3*28*28)
        images = images.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_function(output, true_labels)
        loss.backward()
        optimizer.step()

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time:{}' 
                   .format(epoch_num+1, max_epoch, index+1, total_step, loss.item(), time.time()-iter_data_time))

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch_num, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            
            save_networks(model,save_suffix,save_dir)
    epoch_time = time.time()-epoch_start_time
    print("End of epoch{},Use time:{}".format(epoch_num+1,epoch_time))
    print("Saving the latest model....")
    save_networks(model,epoch='latest',save_dir=save_dir)
    return epoch_time

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of FC network"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset_name')
    parser.add_argument('--name', type=str, default='l3_h_2048', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--dataroot', type=str, default='datas/dataset', help='path of dataset')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch size')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epoch')
    parser.add_argument('--results',type=str, default='./results',help='results are saved here')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--num_val', type=int, default=5, help='start valication when epoch reach ...')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--save_latest_freq', type=int, default=200000, help='frequency of saving the latest results')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='float("inf"),Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_args()
    opt.save_by_iter = True
    if opt is None:
      exit()
    #make checkpoints dir 
    expr_dir = [os.path.join(opt.checkpoints_dir, opt.name),os.path.join(opt.results, opt.name)]
    mkdirs(expr_dir)

    # if len(opt.gpu_ids) > 0:
    #     torch.cuda.set_device(opt.gpu_ids[0])

    # 构建数据集，参数和值需自行查阅相关资料补充。
    dataset_train = HandWritingNumberRecognize_Dataset(opt,"train")

    dataset_val = HandWritingNumberRecognize_Dataset(opt,"val")

    dataset_test = HandWritingNumberRecognize_Dataset(opt,"test")

    # 构建数据加载器，参数和值需自行完善。
    data_loader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=opt.batch_size,
            shuffle=True,)

    data_loader_val = DataLoader(
            dataset=dataset_val,
            batch_size=opt.batch_size,
            shuffle=True,
    )

    data_loader_test = DataLoader(
            dataset=dataset_test,
            batch_size=opt.batch_size,
            shuffle=False,
    )
    #set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    gpu_ids = opt.gpu_ids
    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

    # 初始化模型对象，可以对其传入相关参数
    model = HandWritingNumberRecognize_Network(opt)
    
    # stat(model = model,input_size=(1,1,3*28*28))
    if len(gpu_ids)>0:
        model.to(gpu_ids[0])
        
        model = torch.nn.DataParallel(model, gpu_ids)
    summary(model.module,(3*28*28,))
    # 损失函数设置
    loss_function = nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置

    # 优化器设置
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # torch.optim中的优化器进行挑选，并进行参数设置
    max_epoch = opt.epoch  # 自行设置训练轮数
    num_val = opt.num_val  # 经过多少轮进行验证
    acc_his = []
    time_his = []
    # 然后开始进行训练
    for epoch in range(max_epoch):
        time_his.append(train(epoch,data_loader_train=data_loader_train,opt = opt))
        # 在训练数轮之后开始进行验证评估
        if epoch % num_val == 0:
            acc_his.append(validation(data_loader_val=data_loader_val))
            
    his = {'acc':acc_his,'time':time_his}
    save_his(his,opt)
    # 自行完善测试函数，并通过该函数生成测试结果
    alltest(data_loader_test)
