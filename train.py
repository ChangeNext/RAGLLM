import os
import torch
import json
import random
import numpy as np
from datetime import datetime
from peft import get_peft_model, LoraConfig
from dataset.text import SentenceEmbeddings
from src.model.model import TeMoLLM, TeMoLLM_Config
from dataset.data import (
    AMASSMotionLoader, 
    Normalizer, 
    TextMotionDataset, 
    VQMotionDataset
)
from options import option
import src.util.utils_model as utils_model
from torch.utils.tensorboard import SummaryWriter
###### ----  seed  ---- #####
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

args = option.get_args_parser()
model_name = os.path.basename(args.llm_models)
date_time_str = datetime.now().strftime("%Y%m%d_%H%M")
args.save_dir = os.path.join(args.save_dir, f"{args.batch_size}_{args.threshold}_{args.max_length}_{args.shared_emb_dim}_{model_name}_{args.bf16}_{date_time_str}")
os.makedirs(args.save_dir, exist_ok=True)
###### ---- Logger ---- ######
logger = utils_model.get_logger(args.save_dir)
writer = SummaryWriter(args.save_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

lora_config = {
    'r': args.lora_rank,
    'bias' : args.lora_bias,
    'lora_alpha': args.lora_alpha,
    'lora_dropout': args.lora_dropout,
    'target_modules' : ['q_proj', 'k_proj','o_proj', 'v_proj'],
    'task_type': "CAUSAL_LM",
}

set_seed(args.seed)
model_args = TeMoLLM_Config(lora_config=lora_config)
model_args.freeze_mm = True
model_args.llm_model = args.llm_models
model_args.shared_emb_dim = args.shared_emb_dim
model_args.input_prompt = args.input_prompt
model_args.text_emb_layers = [-1]
model_args.bf16 = args.bf16

temollm = TeMoLLM.from_pretrained(model_kwargs = model_args, logger=logger)

nomalizer = Normalizer(base_dir="./dataset/stats/humanml3d/guoh3dfeats", eps=1e-12)
motion_loader = AMASSMotionLoader(base_dir="./dataset/motions/guoh3dfeats", fps=20.0, normalizer=nomalizer, disable= False, nfeats=263)
sent_emd = SentenceEmbeddings(modelname = "sentence-transformers/all-mpnet-base-v2", path = "./dataset/annotations/humanml3d", preload= True)
cnn_motion_loader = VQMotionDataset(unit_length=2**2)

train_dataset = TextMotionDataset(
    path = './dataset/annotations/humanml3d', 
    split = 'train',
    max_len = args.max_length,
    tokenizer=temollm.model.tokenizer,
    motion_loader = motion_loader,
    cnn_motion_loader=cnn_motion_loader, 
    text_to_sent_emb = sent_emd,
    preload=False,
    return_dict = False
)

val_dataset = TextMotionDataset(
    path = './dataset/annotations/humanml3d', 
    split = 'test',
    max_len = args.max_length,
    tokenizer=temollm.model.tokenizer,
    motion_loader = motion_loader,
    cnn_motion_loader=cnn_motion_loader, 
    text_to_sent_emb = sent_emd,
    preload=False,
    return_dict = True
)

temollm.fit(
    train_ds = train_dataset,
    valid_ds = val_dataset,
    logger = logger,
    threshold = args.threshold,
    batch_size = args.batch_size,
    output_dir = args.save_dir,
    epochs = args.epochs,
    learning_rate = args.lr,
    warmup_steps = args.warmup_steps,
    save_total_limit = 5,
    bf16 = args.bf16,
    lr_scheduler_type = "cosine",
    max_steps=-1,
    weight_decay=0.1,
    logging_steps = 100,
    dataloader_pin_memory=False,
    seed=args.seed,
    do_train=True,
    do_eval=False,
    eval_strategy="no",
    save_strategy="epoch",
    overwrite_output_dir=True,
)