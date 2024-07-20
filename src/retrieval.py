
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from dataset.collate import collate_text_motion
from src.metrics import all_contrastive_metrics, print_latex_metrics
import logging

logger = logging.getLogger(__name__)
# x.T will be deprecated in pytorch
def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

def get_sim_matrix(x, y):
    logits_per_text = x @ y.t()
    return logits_per_text

def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)
        
def save_metric_train(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode) as f:
        f.write("############################################\n")
        f.write(strings)        

# def to_dict(dataset, keyids):
#     all_data = {
#     "tokenized_data": [],
#     "motion_x_dict": [],
#     "text": [],
#     "keyid": [],
#     "sent_emb": [],
#     "motion_cnn": [],
#     "caption_len": []
#     }
#     for keyid in keyids:
#         tokenized_data, motion_x_dict, text, keyidss, sent_emb, motion_cnn, caption_len = dataset.load_keyid(keyid)
    
#         all_data["tokenized_data"].append(tokenized_data)
#         all_data["motion_x_dict"].append(motion_x_dict)
#         all_data["text"].append(text)
#         all_data["keyid"].append(keyidss)
#         all_data["sent_emb"].append(sent_emb)
#         all_data["motion_cnn"].append(motion_cnn)
#         all_data["caption_len"].append(caption_len)
        
#     return all_data

def compute_sim_matrix(model, dataset, keyids, device, batch_size=256):
    device = device
    nsplit = int(np.ceil(len(dataset) / batch_size))
    with torch.no_grad():
        all_data = [dataset.load_keyid(keyid, retrun_dict=True) for keyid in keyids]
        # all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        all_data_splitted = np.array_split(all_data, nsplit)
        latent_texts = []
        latent_motions = []
        sent_embs = []
        for data in tqdm(all_data_splitted, leave=False):
            batch = collate_text_motion(data, device=device, input_dict=True)
            latent_text, latent_motion = model.eval_forward(batch)
            sent_emb = batch["sent_emb"]

            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sent_emb)
            
        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)
        sent_embs = torch.cat(sent_embs)
        sim_matrix = get_sim_matrix(latent_texts, latent_motions)
        
    returned = {
        "sim_matrix": sim_matrix.cpu().numpy(),
        "sent_emb": sent_embs.cpu().numpy()
    }
    return returned


from torch.utils.data import Dataset
from typing import Optional

def retrieval(protocol, dataset, threshold, model, device, save_dir, batch_size, train_mode, logger, nsim_dataset: Optional[Dataset]=None):
    device = device
    protocol = protocol
    threshold_val  = threshold
    assert protocol in ["all", "normal", "threshold", "nsim", "guo", "else"]

    if protocol == "all":
        protocols = ["normal", "threshold", "nsim", "guo"]
    else:
        protocols = [protocol]
    
    model.eval()

    datasets = {}
    results = {}
    metrics_list = []

    for protocol in protocols:
        # Load the dataset if not already
        if protocol not in datasets:
            if protocol in ["normal", "threshold", "guo"]:
                datasets.update(
                    {key: dataset for key in ["normal", "threshold", "guo"]}
                )
            elif protocol == "nsim":
                datasets[protocol] = nsim_dataset
        dataset = datasets[protocol]
        # Compute sim_matrix for each protocol
        if protocol not in results:
            if protocol in ["normal", "threshold"]:
                res = compute_sim_matrix(
                    model, dataset, dataset.keyids, device=device, batch_size=batch_size
                )
                results.update({key: res for key in ["normal", "threshold"]})
                
        result = results[protocol]
        
        # Compute the metrics
        if protocol in ["normal", "threshold", "guo"]:
            sim_matrix = result["sim_matrix"]
            protocol_name = protocol
            if protocol == "threshold":
                emb = result["sent_emb"]
                threshold = threshold_val
                protocol_name = protocol + f"_{threshold}"
            else:
                emb, threshold = None, None
            metrics = all_contrastive_metrics(sim_matrix, emb, threshold=threshold)
        metrics_list.append(metrics)
        print_latex_metrics(metrics, logger)

        metric_name = f"{protocol_name}.yaml"
            
        path = os.path.join(save_dir, metric_name)
        save_metric_train(path, metrics)
        
    
    if train_mode:
        return metrics_list
    else:
        logger.info(f"Testing done, metrics saved in:\n{path}")
        return metrics