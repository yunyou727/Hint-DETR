# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
if __name__ == "__main__":
    # for debug only
    import os, sys

    sys.path.append(os.path.dirname(sys.path[0]))

import json
from pathlib import Path
import random
import os

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms_patch as T
import torchvision.transforms as transforms
from util.box_ops import box_cxcywh_to_xyxy, box_iou
import torch.nn.functional as F
import copy

__all__ = ['build']


def patch_transform():
    normalize = T.Compose1([
        T.ToTensor1(),
        T.Normalize1([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [500, 524, 542, 566, 578, 600, 622, 644, 668, 678, 700]
    max_size = 800

    scales1 = [600, 624, 642, 666, 678, 700, 722, 744, 768, 778, 800]
    max_size1 = 850

    return T.Compose1([
        T.RandomColorJitter(),
        T.RandomHorizontalFlip1(),
        T.RandomSelect1(
            T.RandomResize1(scales, max_size=max_size),
            T.RandomResize1(scales1, max_size=max_size1)

        ),
        normalize])


def bg_transform():
    normalize = T.Compose1([
        T.ToTensor1(),
        T.Normalize1([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales1 = [400, 424, 442, 466, 478, 500, 522, 544, 568, 678, 600]
    max_size1 = 700

    scales = [500, 524, 542, 566, 578, 600, 622, 644, 668, 678, 700]
    max_size = 800

    return T.Compose1([
        T.RandomColorJitter(),
        T.RandomHorizontalFlip1(),
        # T.RandomResize1(scales, max_size=max_size),
        T.RandomSelect1(
            T.RandomResize1(scales, max_size=max_size),
            T.RandomResize1(scales1, max_size=max_size1),

        ),
        normalize])


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, aux_target_hacks=None, patch_transforms=None,
                 bg_transforms=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks
        self.patch_transforms = patch_transforms
        self.bg_transforms = bg_transforms

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)

        numobj = len(target)

        w, h = img.size
        img1 = copy.deepcopy(img)
        to_tensor = transforms.ToTensor()
        img1_tensor = to_tensor(img1)
        for i in range(0, numobj):
            bbox = target[i]['bbox']
            (x, y, sw, sh) = (bbox[0], bbox[1], bbox[2], bbox[3])
            x = int(x)
            y = int(y)
            sw = int(sw)
            sh = int(sh)

            x1, y1, x2, y2 = max(0, x), max(0, y), min(w - 1, x + sw), min(h - 1, y + sh)
            h11, w11 = y2 - y1 + 1, x2 - x1 + 1
            grid = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, h11),
                torch.linspace(-1, 1, w11),
                indexing='ij'
            ), dim=-1).unsqueeze(0)  # (1, h, w, 2)

            # 执行双线性插值
            img_region_tensor = img1_tensor[:, y1:y2 + 1, x1:x2 + 1].unsqueeze(0)  # (1, C, h, w)
            interpolated_region_tensor = F.grid_sample(img_region_tensor, grid, mode='bilinear',
                                                       align_corners=True).squeeze(0)

            # 替换原始图像中的矩形区域
            img1_tensor[:, y1:y2 + 1, x1:x2 + 1] = interpolated_region_tensor

        to_pil = transforms.ToPILImage()
        img1 = to_pil(img1_tensor)
        bg = img1
        bg = self.bg_transforms(bg)

        patchset = []
        num_class = 6
        for i in range(0, num_class):
            patchset.append(0)
        for i in range(0, numobj):
            l = target[i]['category_id'] - 1
            if not torch.is_tensor(patchset[l]):
                bbox = target[i]['bbox']
                (x, y, sw, sh) = (bbox[0], bbox[1], bbox[2], bbox[3])
                patch = img.crop((x, y, x + sw, y + sh))
                if self.patch_transforms is not None:
                    # patch, t = self.patch_transforms(patch, target=None)
                    patch = self.patch_transforms(patch)
                patchset[l] = patch
            if torch.is_tensor(patchset[l]) and random.choice([True, False]):
                bbox = target[i]['bbox']
                (x, y, sw, sh) = (bbox[0], bbox[1], bbox[2], bbox[3])
                patch = img.crop((x, y, x + sw, y + sh))
                if self.patch_transforms is not None:
                    # patch, t = self.patch_transforms(patch, target=None)
                    patch = self.patch_transforms(patch)
                patchset[l] = patch

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, patchset, bg, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        if strong_aug:
            import datasets1.sltransform as SLT

            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        # T.RandomResize(scales2_resize),
                        # T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    # SLT.Rotate(10),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                # # for debug only  
                # SLT.RandomCrop(),
                # SLT.LightingNoise(),
                # SLT.AdjustBrightness(2),
                # SLT.AdjustContrast(2),
                # SLT.Rotate(10),
                normalize,
            ])

        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    # assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "train_reg": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json'),
    }

    # add some hooks to datasets1
    aux_target_hacks_list = None
    img_folder, ann_file = PATHS[image_set]

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False

    try:
        fix_size = args.fix_size
    except:
        fix_size = False
    dataset = CocoDetection(img_folder, ann_file,
                            transforms=make_coco_transforms(image_set, fix_size=fix_size, strong_aug=strong_aug,
                                                            args=args),
                            return_masks=args.masks,
                            aux_target_hacks=aux_target_hacks_list,
                            patch_transforms=patch_transform(),
                            bg_transforms=bg_transform()
                            )

    return dataset
