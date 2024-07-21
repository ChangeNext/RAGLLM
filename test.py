import os
import json
import torch 
from src.model.model import TeMoLLM, TeMoLLM_Config
from src.retrieval import retrieval
from dataset.text import SentenceEmbeddings
from dataset.data import (
    AMASSMotionLoader, 
    Normalizer, 
    TextMotionDataset, 
    VQMotionDataset
)
from collections import namedtuple
from src.model.model import load_TeMoLLM

nomalizer = Normalizer(base_dir="./dataset/stats/humanml3d/guoh3dfeats", eps=1e-12)
motion_loader = AMASSMotionLoader(base_dir="./dataset/motions/guoh3dfeats", fps=20.0, normalizer=nomalizer, disable= False, nfeats=263)
sent_emd = SentenceEmbeddings(modelname = "sentence-transformers/all-mpnet-base-v2", path = "./dataset/annotations/humanml3d", preload= True)
cnn_motion_loader = VQMotionDataset(unit_length=2**2)

if __name__ == "__main__":
    
    model_dir = "/data/jw/motion/TextMotionRetrieval/TMR_LLM/result/train/64_0.8_32_256_20240720_1356"
    basename = os.path.basename(model_dir)
    save_dir = f"./result/test/{basename}"

    temollm = load_TeMoLLM(model_dir)

    nsim_dataset = TextMotionDataset(
        path = './dataset/annotations/humanml3d', 
        split = 'nsim_test',
        max_len = 32,
        tokenizer=temollm.model.tokenizer,
        motion_loader = motion_loader,
        cnn_motion_loader=cnn_motion_loader, 
        text_to_sent_emb = sent_emd,
        preload=False,
        )

    test_dataset = TextMotionDataset(
        path = './dataset/annotations/humanml3d', 
        split = 'test',
        max_len = 32,
        tokenizer=temollm.model.tokenizer,
        motion_loader = motion_loader,
        cnn_motion_loader=cnn_motion_loader, 
        text_to_sent_emb = sent_emd,
        preload=False,
        )

    metric = retrieval(protocol="all", dataset=test_dataset, threshold=0.95, model=temollm.model, device="cuda", save_dir = save_dir, train_mode = False, batch_size=137, logger=None, nsim_dataset=nsim_dataset)