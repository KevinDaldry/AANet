import os
import numpy as np
import torch.nn as nn
import torch.autograd
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from pathlib import Path
from SegmentationMetrics import SegmentationMetric
working_path = os.path.dirname(os.path.abspath(__file__))
# Data and model choose
###############################################
from Modules.sagnet import SAGNet
NET_NAME = 'SAGNet'
DATA_NAME = 'SYSU'
###############################################
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
###############################################
# Test Params
args = {
    'pred_batch_size': 64,
    'gpu': True,
    'pred_dir': os.path.join(working_path, 'CDTask_predictions', DATA_NAME, NET_NAME, 'preds'),
    'gt_dir': os.path.join(working_path, 'CDTask_predictions', DATA_NAME, NET_NAME, 'gts'),
    'load_path': os.path.join(working_path, 'best_weights', DATA_NAME, 'sysu_best_8257_7031.pth')
}
###############################################

if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['gt_dir']): os.makedirs(args['gt_dir'])

ST_COLORMAP = [[0, 0, 0], [255, 255, 255]]

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def main():

    net = SAGNet(3, 2, 128).to('cuda')
    net.load_state_dict(torch.load(args['load_path']), strict=False)

    pred_set = DS(folder=r'./SYSU_test/label/', reference=r'./SYSU_test/A/', reference2=r'./SYSU_test/B/')
    pred_loader = DataLoader(pred_set, batch_size=args['pred_batch_size'], shuffle=False)

    prediction(pred_loader, net)
    print('Prediction finished.')


def prediction(pred_loader, net):
    net.eval()
    torch.cuda.empty_cache()
    count = 0
    ep = 0
    SegMetric = SegmentationMetric(numClass=2)

    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B, label = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            label = label.cuda().long()

        with torch.no_grad():
            out_change = net(imgs_A, imgs_B)
        ep += 1

        pred = torch.argmax(F.softmax(out_change, dim=1), dim=1).squeeze().long()
        SegMetric.addBatch(pred.to('cpu'), label.to('cpu'))

        B, _, _ = pred.shape
        for i in range(B):
            count += 1
            pred_color = pred[i, :, :].to('cpu').numpy()
            label_color = label[i, :, :].to('cpu').numpy()
            cv2.imwrite(os.path.join(args['pred_dir'], NET_NAME + f'_pred_{vi}_{i}.png'),
                        pred_color * 255)
            cv2.imwrite(os.path.join(args['gt_dir'], NET_NAME + f'_label_{vi}_{i}.png'),
                        label_color * 255)

        print(f'Prediction sequence {vi} saved! now {count} pics')

    acc = SegMetric.pixelAccuracy()
    recall = SegMetric.recall()
    precision = SegMetric.precision()
    f1 = SegMetric.F1()
    iou = SegMetric.meanIntersectionOverUnion()

    print(
        f'{count} pred result: acc: {acc * 100: .2f}%, recall: {recall * 100: .2f}%, precision: {precision * 100: .2f}%, '
        f'f1 score: {f1 * 100: .2f}%, iou: {iou * 100: .2f}%')

    return acc, iou, f1


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
