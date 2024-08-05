import os
from os.path import join as pjoin
import random
import numpy as np
import codecs as cs
import json
import codecs as cs
import torch.nn as nn

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Union, Optional
from .collate import collate_text_motion




from tqdm import tqdm

class VQMotionDataset:
    def __init__(
        self, window_size=64, unit_length = 4, id_list = None):
        
        self.window_size = window_size
        self.unit_length = unit_length
        
        self.data_root = '/data/jw/dataset/HumanML3D'
        self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
        self.text_dir = pjoin(self.data_root, "texts")
        self.joints_num = 22
        self.max_motion_length = 196
        self.meta_dir = '/data/jw/dataset/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        self.id_list = id_list
        
        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        

        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

        # self.mean = mean
        # self.std = std
    
    # def filter_id(self, id_list):
    #     for name in tqdm(id_list):
    #         motion = np.load(pjoin(self.motion_dir, name + '.npy'))
    #         if motion.shape[0] < self.window_size:
    #             id_list.remove(name)
    #         # self.lengths.append(motion.shape[0] - self.window_size)
    #         # self.data_list[name] = motion
        
        return id_list
    
    def __call__(self, key_id):
        # try:
        motion = np.load(pjoin(self.motion_dir, key_id + '.npy'))

        if motion.shape[0] > self.window_size:
            downsample = nn.AdaptiveAvgPool1d(self.window_size)
            motion = torch.tensor(motion).permute(1, 0).unsqueeze(0)
            motion = downsample(motion)
            motion = motion.permute(0, 2, 1).squeeze(0)
        elif motion.shape[0] < self.window_size:
            upsample = nn.Upsample(size=self.window_size, mode='linear', align_corners=False)
            motion = torch.tensor(motion).permute(1, 0).unsqueeze(0)
            motion = upsample(motion)
            motion = motion.permute(0, 2, 1).squeeze(0)
        else:
            motion = torch.tensor(motion)
        # motion = motion[idx:idx+self.window_size]
        motion = (motion - self.mean) / self.std
        # motion = motion.to(torch.bfloat16)
        return motion
        # except:
        #     pass    
            
    def inv_transform(self, data):
        return data * self.std + self.mean
        
        
class AMASSMotionLoader:
    def __init__(
        self, base_dir, fps, normalizer=None, disable: bool = False, nfeats=None
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.normalizer = normalizer
        self.disable = disable
        self.nfeats = nfeats
        self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.upsample = nn.
    def __call__(self, path, start, end, split):
        if self.disable:
            return {"x": path, "length": int(self.fps * (end - start))}

        begin = int(start * self.fps)
        end = int(end * self.fps)
        
        if path not in self.motions:
            motion_path = os.path.join(self.base_dir, path + ".npy")
            motion = np.load(motion_path)
            motion = torch.from_numpy(motion).to(torch.float)
            
            if self.normalizer is not None:
                motion = self.normalizer(motion)
                
            self.motions[path] = motion
        
        if split == "train":
            coin = np.random.choice([False, False, True])
            if coin:
                coin2 = np.random.choice([True, False])
                if coin2:
                    end = end - random.randint(0,2)        
                else:
                    start = start + random.randint(0,2)

            motion = self.motions[path][begin:end]
            x_dict = {"x": motion, "length": len(motion)}
            
            return x_dict
            
        else:
            motion = self.motions[path][begin:end]
            x_dict = {"x": motion, "length": len(motion)}

            return x_dict
    
class Normalizer:
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        self.base_dir = base_dir
        self.mean_path = os.path.join(base_dir, "mean.pt")
        self.std_path = os.path.join(base_dir, "std.pt")
        self.eps = eps

        self.disable = disable
        if not disable:
            self.load()

    def load(self):
        self.mean = torch.load(self.mean_path)
        self.std = torch.load(self.std_path)

    def save(self, mean, std):
        os.makedirs(self.base_dir, exist_ok=True)
        torch.save(mean, self.mean_path)
        torch.save(std, self.std_path)

    def __call__(self, x):
        if self.disable:
            return x
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        x = x * (self.std + self.eps) + self.mean
        return x
    
    
def read_split(path, split):
    split_file = os.path.join(path, "splits/" +split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return json.loads(ff.read())
    

def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
        
class TextMotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        motion_loader = None,
        cnn_motion_loader = None,
        text_to_sent_emb = None,
        tokenizer = None,
        max_len: int = 32,
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        return_dict: bool = False,
    ):  
        self.retrun_dict = return_dict
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cnn_motion_loader = cnn_motion_loader
        self.input_prompt = None

        self.collate_fn = collate_text_motion
        self.split = split
        self.keyids = read_split(path, split)
        
        self.text_to_sent_emb = text_to_sent_emb
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        self.annotations = load_annotations(path)
        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test" not in split:
            self.annotations = self.filter_annotations(self.annotations)

        self.is_training = split == "train" 
        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]

        self.nfeats = self.motion_loader.nfeats
        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue
            
    def __len__(self):
        return len(self.keyids)

    def load_keyid(self, keyid):
        annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))
        annotation = annotations["annotations"][index]

        text = annotation["text"]

        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation["start"],
            end=annotation["end"],
            split = self.split
        )
        motion_cnn = self.cnn_motion_loader(keyid)
        sent_emb = self.text_to_sent_emb(text)
        
        eos_token = self.tokenizer.eos_token
        text += eos_token
        tokenized_data = self.tokenizer(
            text, 
            return_tensors='pt',
            padding = 'max_length',
            truncation=True,
            max_length = self.max_len)


        # if tokenized_data.input_ids[0][-1] not in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
        #     tokenized_data.input_ids[0][-1] = self.tokenizer.eos_token_id
        if tokenized_data.input_ids[0][-1] not in [self.tokenizer.eos_token_id]:
            tokenized_data.input_ids[0][-1] = self.tokenizer.eos_token_id
            
        caption_len = tokenized_data.attention_mask[0].sum()

        if self.retrun_dict:
            output = {
                'token' : tokenized_data,
                "motion_x_dict": motion_x_dict,
                "text": text,
                "keyid": keyid,
                "sent_emb": sent_emb,
                'cnn_motion' : motion_cnn,
                'caption_len': caption_len,
            }
            return output
        else:
            return tokenized_data, motion_x_dict, text, keyid, sent_emb, motion_cnn, caption_len
    
    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)
    
    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations