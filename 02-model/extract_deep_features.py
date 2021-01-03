''' Save deep features '''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import scipy.io as sio

from models import *
from utils import progress_bar, Visualizer, lr_schedule
from resnetdataset import *
from collections import OrderedDict
from easydict import EasyDict as edict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cfg = edict({
    'version': '20201127',
    'model': 'resnet50',

    'data_root': 'G:/DataBase/02_ZS_HCC_pathological/03-code/03-model',  # root directory to train/test dataset.
    'data_version': 'size256_down2',

    'lr': 1e-3,
    'restore': 'G:/DataBase/02_ZS_HCC_pathological/03-code/03-model/checkpoint/20201127ckpt.pth'
    })

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_patch_acc = 0  # best test accuracy
best_patient_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# trainset = RESNETdataset(cfg.data_root, cfg.data_version,  (224, 224), 'train', augmentation=False)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False)
# valiset = RESNETdataset(cfg.data_root, cfg.data_version, (224, 224), 'test', augmentation=False)
# valiloader = torch.utils.data.DataLoader(valiset, batch_size=4, shuffle=False)
trainset = RESNETdataset(cfg.data_root, cfg.data_version,  (256, 256), 'train', augmentation=True)    
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
valiset = RESNETdataset(cfg.data_root, cfg.data_version, (256, 256), 'test', augmentation=False)
valiloader = torch.utils.data.DataLoader(valiset, batch_size=16, shuffle=True)

# Model
model_zoo = {'resnet50':    ResNet50(num_classes=4),
            'resnet18':     ResNet18(num_classes=2),
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
    checkpoint = torch.load(cfg.restore)
    net.load_state_dict(checkpoint['net'])
    best_patch_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('restore model from epoch: %d, accuracy: %.2f'%(start_epoch, best_patch_acc))

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# visualizer = Visualizer()

# decay = [40, 60, 80]
# Training
# def train(epoch, lr):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     lr = lr_schedule(lr, epoch, decay, 0.5)
#     optimizer = optim.Adam(net.parameters(), lr=lr)
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, batch_data in enumerate(trainloader):
#         inputs = batch_data['data'].to(device)
#         targets = batch_data['label'].to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#         progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | Acc: %.2f%% (%d/%d)'
#             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
#     return train_loss/(batch_idx+1), 100.*correct/total, lr

def extract_features(net, dataloader):
    net.eval()
    with torch.no_grad():
        # first train data
        cur_id = []
        dst_path = 'E:/ZS_pathological/03-us-pathological-classification/his_deep_features'
        for batch_idx, batch_data in enumerate(dataloader):
            inputs = batch_data['data'].to(device)
            targets = batch_data['label'].to(device)
            idxs = batch_data['idx']
            # slices = batch_data['slice']
            _, features = net(inputs)

            for i, idx in enumerate(idxs):
                if cur_id != idx:
                    dst_full = dst_path
                    if not os.path.exists(dst_full):
                        os.mkdir(dst_full)
                    cur_id = idx
                    count = 0
                feature = features[i].cpu().numpy()
                label = targets[i].cpu().numpy()
                save_full = os.path.join(dst_full, idx + '_' +str(count) + '.mat')
                count += 1
                sio.savemat(save_full, {'features': feature, 'label': label})


def save_result(net, dataloader):
    net.eval()
    with torch.no_grad():
        # first train data
        cur_id = []
        all_predict = []
        all_target  = []
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_idx / 100 == 0:
                print(str(batch_idx))
            inputs = batch_data['data'].to(device)
            targets = batch_data['label'].to(device)
            idxs = batch_data['idx']
            # slices = batch_data['slice']
            outputs = net(inputs)

            _, predicted = outputs.max(1)  # (bs, 1)

            all_predict.append(predicted.cpu().numpy())
            all_target.append(targets.cpu().numpy())

        sio.savemat('predict_result', {'test_label': np.array(all_target), 'predict': np.array(all_predict)})

save_result(net, valiloader)
# extract_features(net, trainloader)

# def vali(epoch):
#     global best_patch_acc, best_patient_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     idx_score_dict = {}
#     with torch.no_grad():
#         for batch_idx, batch_data in enumerate(valiloader):
#             inputs = batch_data['data'].to(device)
#             targets = batch_data['label'].to(device)
#             idxs = batch_data['idx']
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#             # caculate for each patient
#             for i, idx in enumerate(idxs):
#                 if idx not in idx_score_dict:
#                     idx_score_dict[idx] = []
#                 batch_pred = list(np.array(predicted.eq(targets).cpu()))  # [0, 0, ..., 1]
#                 idx_score_dict[idx].append(batch_pred[i])


#             progress_bar(batch_idx, len(valiloader), 'Loss: %.2f | Acc: %.2f%% (%d/%d)'
#                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
#     # print patient result
#     patient_total = 0
#     patient_correct = 0
#     for key in list(idx_score_dict.keys()):
#         patient_total += 1
#         score = np.mean(idx_score_dict[key])
#         if score > 0.5:
#             patient_correct += 1
#     patient_acc = 100. * patient_correct / patient_total
#     print('********** patient num: %d,  correct prediction: %d, patient accuracy: %.2f%% **********'%(patient_total, patient_correct, patient_acc))

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_patch_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/%sckpt.pth'%cfg.version)
#         best_patch_acc = acc

#     if patient_acc > best_patient_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/best_patient_acc_%sckpt.pth'%cfg.version)
#         best_patient_acc = patient_acc
#     return test_loss/(batch_idx+1), 100.*correct/total

# lr = cfg.lr
# for epoch in range(start_epoch, start_epoch+cfg.epoch):
#     train_loss, train_acc, lr = train(epoch, lr)
#     vali_loss, vali_acc = vali(epoch)
#     value_ret = {'train_loss': train_loss, 'train_acc': train_acc,
#                 'vali_loss': vali_loss, 'vali_acc': vali_acc, 'learning_rate*5e4': lr*5e4 }
#     visualizer.plot_current_losses(epoch, 0, value_ret)
