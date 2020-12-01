import torch
import os
import numpy as np

import torchvision.transforms as transforms
import mytransforms as mytransforms
from constant import tusimple_row_anchor, culane_row_anchor
from dataset import LaneClsDataset, LaneTestDataset, LaneGenPseudoDataset
from datasetUtils import get_partial_dataset, split_dataset


def get_train_loader(batch_size, data_root, griding_num, dataset, use_aux,
                     distributed, num_lanes, proportion=1, split=False, split_proportion=0.5, load_name=False, pin_memory=False, num_workers=4):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((288, 800)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])
    if dataset == 'CULane':
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(
                                           data_root, 'list/train_gt.txt'),
                                       img_transform=img_transform, target_transform=target_transform,
                                       simu_transform=simu_transform,
                                       segment_transform=segment_transform,
                                       row_anchor=culane_row_anchor,
                                       griding_num=griding_num, use_aux=use_aux, num_lanes=num_lanes, load_name=load_name)
        cls_num_per_lane = 18
    elif dataset == 'CULane-pseudo':
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(
                                           data_root, 'train_pseudo_gt.txt'),
                                       img_transform=img_transform, target_transform=target_transform,
                                       simu_transform=simu_transform,
                                       griding_num=griding_num,
                                       row_anchor=culane_row_anchor,
                                       segment_transform=segment_transform, use_aux=use_aux, num_lanes=num_lanes, load_name=load_name)
        cls_num_per_lane = 18

    elif dataset == 'CULane-no-aug':
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(
                                           data_root, 'list/train_gt.txt'),
                                       img_transform=img_transform, target_transform=target_transform,
                                       simu_transform=None,
                                       griding_num=griding_num,
                                       row_anchor=culane_row_anchor,
                                       segment_transform=segment_transform, use_aux=use_aux, num_lanes=num_lanes, load_name=load_name)
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(data_root, 'train_gt.txt'),
                                       img_transform=img_transform, target_transform=target_transform,
                                       simu_transform=simu_transform,
                                       griding_num=griding_num,
                                       row_anchor=tusimple_row_anchor,
                                       segment_transform=segment_transform, use_aux=use_aux, num_lanes=num_lanes, load_name=load_name)
        cls_num_per_lane = 56
    elif dataset == 'Tusimple-pseudo':
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(
                                           data_root, 'train_pseudo_gt.txt'),
                                       img_transform=img_transform, target_transform=target_transform,
                                       simu_transform=simu_transform,
                                       griding_num=griding_num,
                                       row_anchor=tusimple_row_anchor,
                                       segment_transform=segment_transform, use_aux=use_aux, num_lanes=num_lanes, load_name=load_name)
        cls_num_per_lane = 56

    elif dataset == 'Tusimple-no-aug':
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(
                                           data_root, 'train_gt.txt'),
                                       img_transform=img_transform, target_transform=target_transform,
                                       simu_transform=None,
                                       griding_num=griding_num,
                                       row_anchor=tusimple_row_anchor,
                                       segment_transform=segment_transform, use_aux=use_aux, num_lanes=num_lanes, load_name=load_name)
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    train_dataset = get_partial_dataset(train_dataset, proportion)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    if split:
        train_datasets = split_dataset(train_dataset, split_proportion)
        if distributed:
            samplers = [torch.utils.data.distributed.DistributedSampler(
                ds) for ds in train_datasets]
        else:
            samplers = [torch.utils.data.RandomSampler(
                ds) for ds in train_datasets]
        train_loader = [torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
                        for ds, sampler in zip(train_datasets, samplers)]

    return train_loader, cls_num_per_lane


def get_test_loader(batch_size, data_root, dataset, distributed, proportion=1, num_workers=4):
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset == 'CULane':
        test_dataset = LaneTestDataset(data_root, os.path.join(
            data_root, 'list/test.txt'), img_transform=img_transforms)
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        test_dataset = LaneTestDataset(data_root, os.path.join(
            data_root, 'test.txt'), img_transform=img_transforms)
        cls_num_per_lane = 56

    test_dataset = get_partial_dataset(test_dataset, proportion)

    if distributed:
        sampler = SeqDistributedSampler(test_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return loader


def get_gen_pseudo_loader(batch_size, data_root, dataset, distributed, num_workers=4):
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset == 'CULane':
        test_dataset = LaneGenPseudoDataset(data_root, os.path.join(
            data_root, 'list/train_pseudo_gt.txt'), img_transform=img_transforms)
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        test_dataset = LaneGenPseudoDataset(data_root, os.path.join(
            data_root, 'train_pseudo_gt.txt'), img_transform=img_transforms)
        cls_num_per_lane = 56

    if distributed:
        sampler = SeqDistributedSampler(test_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return loader


class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    '''
    Change the behavior of DistributedSampler to sequential distributed sampling.
    The sequential sampling helps the stability of multi-thread testing, which needs multi-thread file io.
    Without sequentially sampling, the file io on thread may interfere other threads.
    '''

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank *
                          self.rank: num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)
