import argparse
import logging
import os

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import transforms as T
from dataloader import TinyVocDataset
from engine import train_one_epoch, evaluate, my_eval
import utils

logger = logging.getLogger(__name__)


def get_args():
    """ Argument parser for traing hyperparameter"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str,
                        default='trained_model/current.pkl', help='model.pth path(s)')
    parser.add_argument('--dataset', type=str,
                        default='dataset', help='model.pth path(s)')
    parser.add_argument('--optim', default='SGD',
                        choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lr', '--learning_rate', dest='lr', default=1e-2,
                        type=float, help='optimizer')
    parser.add_argument('-b', '--batch_size', dest='bs',
                        default=4, help='batch size', type=int)
    parser.add_argument('--nr_epoch', default=100,
                        help='number of epoch', type=int)
    parser.add_argument('--start_epoch', default=0,
                        help='start epoch', type=int)
    parser.add_argument('--nr_worker', default=5,
                        help='number of dataloader worker', type=int)
    parser.add_argument('--nr_class', default=21,
                        help='number of categories', type=int)
    parser.add_argument('--eval', action='store_true',
                        help='if set, it will only forwarding the whole training dataset and show the evaluating result')
    parser.add_argument('--test', action='store_true',
                        help='if set, it will only forwarding the whole testing data and show the evaluating result')
    parser.add_argument('--save_json', action='store_true',
                        help='make submission file')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained model on coco dataset')

    opt = parser.parse_args()
    opt.device = torch.device('cuda') if torch.cuda.is_available()\
        else torch.device('cpu')
    print(opt)
    return opt


def get_model(pretrained, num_classes):
    """ To get the maskRcnn model
    Note: TBD, change the backbone model pfn and other modules
    :param: pretrained(bool), if set, the model we load the model weights 
            pretrained on pascal dataset
    :param: num_classes(int), it should be set to N(catagerories) +1(background)
    """
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
    #                                                 output_size=7,
    #                                                 sampling_ratio=2)
    if pretrained:
        print(">>>>>Use pretrained model<<<<<")
        model = maskrcnn_resnet50_fpn(pretrained=True, num_classes=21)
    else:
        model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=21)
    return model


def get_transform(train):
    """ To get the transforms object
    :param: train(bool), if set, it will apply all augmented methods;
            otherwise, only to_tensor and normalize will be used.
    """
    transforms = []
    if train:
        transforms.append(T.ColorJitter(0.2, 0.2, 0.2, 0.05))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train(opt):
    # prepare dataloader
    train_dataset = TinyVocDataset(
        'dataset', transforms=get_transform(train=True), train=True)
    val_dataset = TinyVocDataset(
        'dataset', transforms=get_transform(train=False), train=True)
    nr_train = round(0.8*len(train_dataset))
    indices = torch.randperm(len(train_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:nr_train])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[nr_train:])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.bs, shuffle=True, num_workers=opt.nr_worker, collate_fn=utils.collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.nr_worker, collate_fn=utils.collate_fn)

    # prepare model
    model = get_model(pretrained=opt.pretrained, num_classes=opt.nr_class)
    if os.path.isfile(opt.weight):
        print("load model weight, {}".format(opt.weight))
        model.load_state_dict(torch.load(opt.weight))
        print("model loaded!")
    model.to(opt.device)

    # construct an optimizer and a learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    if opt.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=0.0005)
    elif opt.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=opt.lr, momentum=0.5, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.8)

    best = float('inf')
    for epoch in range(opt.start_epoch, opt.nr_epoch):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader,
                        opt.device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if epoch % 10 == 0:
            evaluate(model, val_loader, device=opt.device)

        eval_res = my_eval(model, val_loader, device=opt.device)
        print(">>>>>>evalutated result, totoal loss = {}<<<<<<".format(eval_res))
        if (eval_res < best):
            torch.save(model.state_dict(),
                       'trained_model/epoch_{}_{}.pkl'.format(epoch, eval_res))
            torch.save(model.state_dict(), 'trained_model/current.pkl')
            best = eval_res


def eval_(opt):
    """ Show evaluated result of training data
    forwarding training datasets and shows the evaluated result measured by cocoeval-tool
    """
    dataset = TinyVocDataset(
        'dataset', transforms=get_transform(False), train=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.nr_worker, collate_fn=utils.collate_fn)

    model = get_model(pretrained=False, num_classes=opt.nr_class)
    if os.path.isfile(opt.weight):
        print("load model weight, {}".format(opt.weight))
        model.load_state_dict(torch.load(opt.weight))
        print("model loaded!")
    model.to(opt.device)
    evaluate(model, dataloader, device=opt.device)


def test(opt):
    """For warding the test data"""
    import matplotlib.pyplot as plt
    from pycocotools.coco import COCO
    import numpy as np
    from utils import binary_mask_to_rle
    from PIL import Image
    from torchvision.transforms import functional as F
    import json

    # prepare model
    model = get_model(pretrained=False, num_classes=opt.nr_class)
    if os.path.isfile(opt.weight):
        print("load model weight, {}".format(opt.weight))
        model.load_state_dict(torch.load(opt.weight))
        print(">>>>>>> model loaded! <<<<<<<<")
    model.to(opt.device).eval()

    # prepare test data
    coco = COCO(os.path.join(opt.dataset, "test.json"))
    # coco = COCO(os.path.join(opt.dataset, "pascal_train.json"))
    img_dir = 'dataset/images/'

    if opt.save_json:
        coco_dt = []
        for imgid in coco.imgs:
            img_path = img_dir + coco.loadImgs(ids=imgid)[0]['file_name']
            print(img_path)
            img = F.to_tensor(Image.open(img_path).convert('RGB'))
            with torch.no_grad():
                output = model([img.to(opt.device)])[0]
            masks = output['masks'].cpu().numpy()
            categories = output['labels'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            # If any objects are detected in this image
            n_instances = len(scores)
            if (len(categories) > 0):
                for i in range(n_instances):  # Loop all instances
                    # if scores[i] < 0.3:
                    #     continue
                    pred = {}
                    pred['image_id'] = imgid
                    pred['category_id'] = int(categories[i])
                    # save binary mask to RLE, e.g. 512x512 -> rle
                    b_mask = np.where(masks[i][0] > 0.5, 1, 0)  # .astype(int)
                    pred['segmentation'] = binary_mask_to_rle(b_mask)
                    pred['score'] = float(scores[i])
                    coco_dt.append(pred)
        with open("submittions/0856126.json", "w") as f:
            json.dump(coco_dt, f)


if __name__ == '__main__':
    opt = get_args()
    if opt.test:
        test(opt)
    elif opt.eval:
        eval_(opt)
    else:
        train(opt)
