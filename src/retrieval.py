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

def compute_sim_matrix(model, dataset, keyids, device, batch_size=256, bf16=False):
    device = device
    nsplit = int(np.ceil(len(dataset) / batch_size))
    with torch.no_grad():
        # all_data = [dataset.load_keyid(keyid, retrun_dict=True) for keyid in keyids]
        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
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
        if bf16:
            sim_matrix = sim_matrix.to(torch.float32).cpu().numpy()
        else:
            sim_matrix = sim_matrix.cpu().numpy()

    returned = {
        "sim_matrix": sim_matrix,
        "sent_emb": sent_embs.cpu().numpy()
    }
    return returned


from torch.utils.data import Dataset
from typing import Optional

def retrieval(protocol, dataset, threshold, model, device, save_dir, batch_size, train_mode, logger, nsim_dataset, bf16):
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
                    model, dataset, dataset.keyids, device=device, batch_size=batch_size, bf16=bf16
                )
                results.update({key: res for key in ["normal", "threshold"]})
            elif protocol == "nsim":
                res = compute_sim_matrix(
                    model, dataset, dataset.keyids, device=device,batch_size=batch_size ,bf16=bf16
                )
                results[protocol] = res
            elif protocol == "guo":
                keyids = sorted(dataset.keyids)
                N = len(keyids)

                # make batches of 32
                idx = np.arange(N)
                np.random.seed(0)
                np.random.shuffle(idx)
                idx_batches = [
                    idx[32 * i : 32 * (i + 1)] for i in range(len(keyids) // 32)
                ]

                # split into batches of 32
                # batched_keyids = [ [32], [32], [...]]
                results["guo"] = [
                    compute_sim_matrix(
                        model,
                        dataset,
                        np.array(keyids)[idx_batch],
                        device=device,
                        batch_size=batch_size,
                        bf16=bf16
                    )
                    for idx_batch in idx_batches
                ]

        result = results[protocol]

        # Compute the metrics
        if protocol == "guo":
            all_metrics = []
            for x in result:
                sim_matrix = x["sim_matrix"]
                metrics = all_contrastive_metrics(sim_matrix, rounding=None)
                all_metrics.append(metrics)

            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = round(
                    float(np.mean([metrics[key] for metrics in all_metrics])), 2
                )

            metrics = avg_metrics
            protocol_name = protocol
        else:
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

        os.makedirs(save_dir, exist_ok=True)            
        path = os.path.join(save_dir, metric_name)
        save_metric_train(path, metrics)
        
    model.train()
    if train_mode:
        return metrics_list
    else:
        print(f"Testing done, metrics saved in:\n{save_dir}")
        return metrics