import os
import random
import numpy as np
import torch.nn as nn
import torch.autograd
import cv2
from pathlib import Path
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from SegmentationMetrics import SegmentationMetric
working_path = os.path.dirname(os.path.abspath(__file__))
###############################################
from Modules.AANet import AANet
NET_NAME = 'AANet'
DATA_NAME = 'SYSU'
###############################################
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
###############################################
# Train Params
'''
All Params for Training can be adjusted here in {args}.
If you want to change the optimizer or lr_decay strategy, plz announce your own and replace ours later in main() func.

lr: init learning rate, float(advised between [0.01, 0.1] when using SGD)
epochs: training epochs, int
gpu: use gpu or not, bool
lr_decay_freq: iteration gap to reduce lr, int
lr_decay_power: weight for reducing lr, float(advised between [0, 1])
weight_decay: weight for param decay, float(advised between [0, 1])
'''
args = {
    'train_batch_size': 32,
    'val_batch_size': 32,
    'lr': None,
    'epochs': None,
    'gpu': True,
    'lr_decay_freq': None,
    'lr_decay_power': None,
    'weight_decay': None,
    'momentum': 0.9,
    'print_freq': 100,
    'predict_step': 5,
    'pred_dir': os.path.join(working_path, 'CDTask_results', DATA_NAME, NET_NAME),
    'chkpt_dir': os.path.join(working_path, 'CDTask_checkpoints', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'best_weights', DATA_NAME, 'sysu_best_8257_7031.pth')}
###############################################

if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])

ST_COLORMAP = [[0, 0, 0], [255, 255, 255]]


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


'''
main() is used to control the whole training/validation process.
Validation is implanted into Training process. Check for the detail in train() and validate().
'''


def main():        

    net = AANet(3, 2, 128).to('cuda')
    # net.load_state_dict(torch.load(args['load_path']), strict=False)

    '''
    folder: gt directory
    reference: t1 directory
    reference2: t2 directory
    '''
    train_set = DS(folder=r'./SYSU_train/label/', reference=r'./SYSU_train/A/', reference2=r'./SYSU_train/B/')
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], shuffle=True)
    val_set = DS(folder=r'./SYSU_val/label/', reference=r'./SYSU_val/A/', reference2=r'./SYSU_val/B/', pre_processed=False)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    aux = DiceLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)

    train(train_loader, net, criterion, optimizer, val_loader, aux, rate=[0.7, 0.3])
    print('Training finished.')


def train(train_loader, net, criterion, optimizer, val_loader, aux_criterion=None, rate=[0.5, 0.5]):
    best_acc = 0
    best_iou = 0.0
    best_loss = 1.0
    all_iters = float(len(train_loader)*args['epochs'])
    curr_epoch = 0
    while True:
        torch.cuda.empty_cache()
        net.train()
        SegMet = SegmentationMetric(numClass=2)
        curr_iter = curr_epoch*len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter+i+1
            if running_iter % args['lr_decay_freq'] == 0:
                adjust_lr(optimizer, running_iter, all_iters)
            imgs_A, imgs_B, label = data
            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                label = label.cuda().long()

            optimizer.zero_grad()
            out_change = net(imgs_A, imgs_B)

            assert out_change.size()[1] == 2, f'output change map size wrong, got {out_change.size()}'

            if aux_criterion is not None and len(rate) == 2:
                loss = rate[0] * criterion(out_change, label) + rate[1] * aux_criterion(out_change, label)
            else:
                loss = criterion(out_change, label)

            loss.backward()
            optimizer.step()

            pred = torch.argmax(F.softmax(out_change, dim=1), dim=1).squeeze().long()
            SegMet.addBatch(pred.to('cpu'), label.to('cpu'))
            acc = SegMet.pixelAccuracy()
            recall = SegMet.recall()
            precision = SegMet.precision()
            f1 = SegMet.F1()
            iou = SegMet.meanIntersectionOverUnion()

            if (i + 1) % args['print_freq'] == 0:
                print(
                    f'Epoch {curr_epoch} loss: {loss: .6f}, acc: {acc * 100: .2f}%, recall: {recall * 100: .2f}%, '
                    f'precision: {precision * 100: .2f}%, f1 score: {f1 * 100: .2f}%, iou: {iou * 100: .2f}%')
                    
        loss_v, acc_v, iou_v, f1_v = validate(val_loader, net, criterion, curr_epoch)
        if (loss_v < best_loss) or (acc_v > best_acc) or (iou_v > best_iou):
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], f'{curr_epoch}'+f'_{loss_v:.2f}_{acc_v:.2f}_{f1_v:.2f}_{iou_v:.2f}.pth'))
        curr_epoch += 1

        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()

    count = 0

    val_loss = 0.0
    ep = 0
    SegMetric = SegmentationMetric(numClass=2)

    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, label = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            label = label.cuda().long()

        with torch.no_grad():
            out_change = net(imgs_A, imgs_B)
            count += 1
            loss = criterion(out_change, label)
        val_loss += loss
        ep += 1

        pred = torch.argmax(F.softmax(out_change, dim=1), dim=1).squeeze().long()
        SegMetric.addBatch(pred.to('cpu'), label.to('cpu'))

        if curr_epoch % args['predict_step'] == 0 and vi % 10 == 0:
            choice = random.choice(range(pred.shape[0]))
            pred_color = pred[choice].to('cpu').numpy()
            label_color = label[choice].to('cpu').numpy()
            cv2.imwrite(os.path.join(args['pred_dir'], NET_NAME + f'_pred_{curr_epoch}_{vi}_{choice}.png'), pred_color * 255)
            cv2.imwrite(os.path.join(args['pred_dir'], NET_NAME + f'_label_{curr_epoch}_{vi}_{choice}.png'), label_color * 255)
            print('Prediction saved!')

    acc = SegMetric.pixelAccuracy()
    recall = SegMetric.recall()
    precision = SegMetric.precision()
    f1 = SegMetric.F1()
    iou = SegMetric.meanIntersectionOverUnion()
    vloss = val_loss / ep

    print(f'{count} acc: {acc * 100: .2f}%, recall: {recall * 100: .2f}%, precision: {precision * 100: .2f}%, '
          f'f1 score: {f1 * 100: .2f}%, iou: {iou * 100: .2f}%, loss: {vloss}')

    return vloss, acc, iou, f1


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()            


def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-8

    def forward(self, pred, target, softmax=True):
        pred = pred.argmax(dim=1) if softmax else nn.Softmax(dim=1)(pred).argmax(dim=1)
        pred = pred.squeeze()
        target = target.squeeze()
        assert pred.size() == target.size(), "the size of predict and target must be equal."
        B = pred.size(0)

        pre = pred.view(B, -1)
        tar = target.view(B, -1)

        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


def CountDiff(image, gt):
    assert image.shape == gt.shape, f'shape cannot match！'
    if len(image.shape) == 2:
        image = image.unsqueeze(dim=0)
    B, H, W = image.shape
    TP = TN = FP = FN = 0
    for k in range(B):
        for i in range(H):
            for j in range(W):
                if image[k][i][j] == gt[k][i][j]:
                    if image[k][i][j] == 1:
                        TP += 1
                    else:
                        TN += 1
                elif image[k][i][j] != gt[k][i][j]:
                    if image[k][i][j] == 1:
                        FP += 1
                    else:
                        FN += 1
    return TP, TN, FP, FN


class DS(Dataset):
    def __init__(
            self,
            folder,
            reference,
            image_size=256,
            reference2=None,
            num_classes=None,
            exts=['jpg', 'jpeg', 'png', 'tiff', 'tif'],
            pre_processed=True
    ):
        super().__init__()
        self.folder = folder
        self.reference = reference
        self.reference2 = reference2 if reference2 is not None else reference
        self.image_size = image_size
        self.classes = num_classes
        self.paths = [p for ext in exts for p in Path(f'{self.folder}').glob(f'**/*.{ext}')]
        self.inf_paths = [p for ext in exts for p in Path(f'{self.reference}').glob(f'**/*.{ext}')]
        self.inf_paths2 = [p for ext in exts for p in Path(f'{self.reference2}').glob(f'**/*.{ext}')]

        assert len(self.paths) == len(self.inf_paths) == len(self.inf_paths2), f'dataset length does not match'

        self.pre_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5)
        ]) if pre_processed else nn.Identity()

        self.transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        infr2 = self.inf_paths2[index]
        infr = self.inf_paths[index]

        img = cv2.imread(str(path), flags=-1)
        img = torch.from_numpy(img).unsqueeze(dim=0).type(torch.float)

        raw = torch.from_numpy(cv2.imread(str(infr))).permute(2, 0, 1).type(torch.float)
        raw2 = torch.from_numpy(cv2.imread(str(infr2))).permute(2, 0, 1).type(torch.float)

        pack = torch.cat([raw, raw2, img], dim=0)
        pack = self.pre_transform(pack)

        raw = pack[: 3, : , : ]
        raw2 = pack[3: 6, : , : ]
        gt = pack[6: , : , : ].squeeze()

        raw = self.transform(raw)
        raw2 = self.transform(raw2)

        assert raw.shape == raw2.shape, f'raw image shape must be same, but have {raw.shape} not equal {raw2.shape}'

        return [raw, raw2, gt]


if __name__ == '__main__':
    main()
