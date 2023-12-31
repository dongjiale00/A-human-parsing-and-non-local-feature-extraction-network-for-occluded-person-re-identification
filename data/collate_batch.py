# encoding: utf-8


import torch


# def train_collate_fn(batch):
#     imgs, pids, _, _, mask_target, _, imgs_2, mask_target_2 = zip(*batch)
#     pids = torch.tensor(pids, dtype=torch.int64)
#     mask_target = torch.tensor(mask_target, dtype=torch.int64)
#     mask_target_2 = torch.tensor(mask_target_2, dtype=torch.int64)
#
#     # print(pids.shape)
#     # print(imgs.shape)
#     # print(mask_target.shape)
#     # print(pids.shape)
#
#     return torch.stack(imgs, dim=0), pids, mask_target, torch.stack(imgs_2, dim=0), mask_target_2

def train_collate_fn(batch):
    imgs, pids, _, _, mask_target, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    mask_target = torch.tensor(mask_target, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, mask_target

def clustering_collate_fn(batch):
    imgs, pids, _, _, _, mask_target_path = zip(*batch)
    return torch.stack(imgs, dim=0), mask_target_path, pids


def val_collate_fn(batch):
    imgs, pids, camids, _, mask_target, _ = zip(*batch)
    mask_target = torch.tensor(mask_target, dtype=torch.int64)
    
    return torch.stack(imgs, dim=0), pids, camids, mask_target