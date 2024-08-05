import torch
from typing import Any, Dict, List, Tuple, Union, Optional
from torch import Tensor
from torch.utils.data import default_collate
from torch.nn.utils.rnn import pad_sequence

def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_x_dict(lst_x_dict: List, *, device: Optional[str] = None, bf16=False) -> Dict:
    x = collate_tensor_with_padding([x_dict["x"] for x_dict in lst_x_dict])
    if device is not None:
        if bf16:
            x = x.to(torch.bfloat16)
        x = x.to(device)
    length = [x_dict["length"] for x_dict in lst_x_dict]
    mask = length_to_mask(length, device=x.device)
    batch = {"x": x, "length": length, "mask": mask}
    return batch

def collate_text_motion(batch, device :Optional[str]=None, input_dict:Optional[bool]=False):
    if input_dict:
        one_el = batch[0]
        keys = one_el.keys()
        x_dict_keys = [key for key in keys if "x_dict" in key]
        other_keys = [key for key in keys if "x_dict" not in key]
        batchs = {key: default_collate([x[key] for x in batch]) for key in other_keys}
        for key, val in batchs.items():
            if isinstance(val, torch.Tensor) and device is not None:
                batchs[key] = val.to(device)
        for key in x_dict_keys:
            batchs[key] = collate_x_dict([x[str(key)] for x in batch], device=device)
        return batchs
    else:
        tokenized_data_batch = [item[0] for item in batch]
        text_batch = [item[2] for item in batch]
        keyid_batch = [item[3] for item in batch]
        sent_emb_batch = [item[4] for item in batch]
        motion_cnn_batch = [item[5] for item in batch]
        caption_len = [item[6] for item in batch]
        collated_batch = {
            'token': default_collate(tokenized_data_batch),
            'text': default_collate(text_batch),
            'keyid': default_collate(keyid_batch),
            'sent_emb': default_collate(sent_emb_batch),
            'cnn_motion': default_collate(motion_cnn_batch),
            'caption_len' : default_collate(caption_len),
        }
        collated_batch["motion_x_dict"] = collate_x_dict([x[1] for x in batch])
        
        return collated_batch