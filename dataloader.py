import os
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO


class TinyVocDataset(object):
    def __init__(self, root, transforms=None, train=True):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(os.path.join(root, 'pascal_train.json')) if train\
            else COCO(os.path.join(root, 'test.json'))
        self.image_ids = self.coco.getImgIds()

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # get metadata
        img_info = self.coco.imgs[image_id]  # dict
        anns = self.coco.imgToAnns[image_id]  # list

        # load images, masks and bboxes
        img_path = os.path.join(self.root, 'images', img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        masks = [self.coco.annToMask(ann) for ann in anns]

        # prepare targets
        boxes = [[ann['bbox'][0], ann['bbox'][1],
                  ann['bbox'][0] + ann['bbox'][2],
                  ann['bbox'][1] + ann['bbox'][3]] for ann in anns]
        labels = [ann['category_id'] for ann in anns]
        area = [ann['area'] for ann in anns]
        iscrowd = [ann['iscrowd'] for ann in anns]

        # convert things to tensor
        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
        target['image_id'] = torch.tensor([image_id], dtype=torch.int64)
        target['area'] = torch.tensor(area)
        target['iscrowd'] = torch.tensor(iscrowd, dtype=torch.uint8)

        # data augmentation
        image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.image_ids)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from train import get_transform
    import matplotlib.patches as patches

    dataset = TinyVocDataset('dataset', transforms=get_transform(True))
    img, target = dataset[1]

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(img.detach().numpy().transpose(1, 2, 0))

    boxes = target['boxes']
    for bbox in boxes:
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()
