import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar, Visualizer, lr_schedule, make_weights_for_balanced_classes
from resnetdataset import *
from collections import OrderedDict
from easydict import EasyDict as edict

cfg = edict({
    'version': '20201127',
    'model': 'resnet50',

    'data_root': 'G:/DataBase/02_ZS_HCC_pathological/03-code/03-model',  # root directory to train/test dataset.
    'data_version': 'size256_down2',

    'lr': 1e-3,
    'restore': False
    })

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_patch_acc = 0  # best test accuracy
best_patient_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
trainset = RESNETdataset(cfg.data_root, cfg.data_version,  (256, 256), 'train', augmentation=True)
# weights = make_weights_for_balanced_classes(trainset, 4)
# weights = torch.DoubleTensor(weights)   
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, sampler=sampler)      
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
valiset = RESNETdataset(cfg.data_root, cfg.data_version, (256, 256), 'test', augmentation=False)
valiloader = torch.utils.data.DataLoader(valiset, batch_size=16, shuffle=True)

# Model
model_zoo = {'resnet50':    ResNet50(num_classes=4),
            'resnet18':     ResNet18(),
            'vgg':          VGG('VGG13', num_classes=4),
            'shufflenetv2': MobileNetV2(),
            'densenet121':  DenseNet121(),
            'googlenet':    GoogLeNet()
            }

print('==> Building %s model..'%cfg.model)
net = model_zoo[cfg.model]
# net = VGG('VGG13', num_classes=4)
# net = ResNet50(num_classes=2)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if cfg.restore:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_patch_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
visualizer = Visualizer()

decay = [4, 8, 12, 16, 20]
# Training
def train(epoch, lr):
    print('\nEpoch: %d' % epoch)
    net.train()
    lr = lr_schedule(lr, epoch, decay, 0.1)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, batch_data in enumerate(trainloader):
        inputs = batch_data['data'].to(device)
        targets = batch_data['label'].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | Acc: %.2f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), 100.*correct/total, lr

def vali(epoch):
    global best_patch_acc, best_patient_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx_score_dict = {}
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(valiloader):
            inputs = batch_data['data'].to(device)
            targets = batch_data['label'].to(device)
            idxs = batch_data['idx']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # caculate for each patient
            for i, idx in enumerate(idxs):
                if idx not in idx_score_dict:
                    idx_score_dict[idx] = []
                batch_pred = list(np.array(predicted.eq(targets).cpu()))  # [0, 0, ..., 1]
                idx_score_dict[idx].append(batch_pred[i])


            progress_bar(batch_idx, len(valiloader), 'Loss: %.2f | Acc: %.2f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # print patient result
    patient_total = 0
    patient_correct = 0
    for key in list(idx_score_dict.keys()):
        patient_total += 1
        score = np.mean(idx_score_dict[key])
        if score > 0.5:
            patient_correct += 1
    patient_acc = 100. * patient_correct / patient_total
    print('********** patient num: %d,  correct prediction: %d, patient accuracy: %.2f%% **********'%(patient_total, patient_correct, patient_acc))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_patch_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/%sckpt.pth'%cfg.version)
        best_patch_acc = acc

    if patient_acc > best_patient_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/best_patient_acc_%sckpt.pth'%cfg.version)
        best_patient_acc = patient_acc
    return test_loss/(batch_idx+1), 100.*correct/total

lr = cfg.lr
for epoch in range(start_epoch, start_epoch+20):
    train_loss, train_acc, lr = train(epoch, lr)
    vali_loss, vali_acc = vali(epoch)
    value_ret = {'train_loss': train_loss, 'train_acc': train_acc,
                'vali_loss': vali_loss, 'vali_acc': vali_acc, 'learning_rate': lr }
    visualizer.plot_current_losses(epoch, 0, value_ret)
