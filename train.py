import os
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


args = option.get_args_parser()
date_time_str = datetime.now().strftime("%Y%m%d_%H%M")
args.save_dir = os.path.join(args.save_dir, f"{args.batch_size}_{args.threshold}_{args.max_length}_{args.shared_emb_dim}_{args.llm_models}{date_time_str}")
lora_config = {
    'r': args.lora_rank,
    'bias' : args.lora_bias,
    'lora_alpha': args.lora_alpha,
    'lora_dropout': args.lora_dropout,
    'target_modules' : ['q_proj', 'k_proj','o_proj', 'v_proj'],
    'task_type': "CAUSAL_LM",
}

model_args = TeMoLLM_Config(lora_config=lora_config)

model_args.llm_model = args.llm_models
model_args.shared_emb_dim = args.shared_emb_dim

temollm = TeMoLLM.from_pretrained(model_kwargs = model_args)

nomalizer = Normalizer(base_dir="./dataset/stats/humanml3d/guoh3dfeats", eps=1e-12)
motion_loader = AMASSMotionLoader(base_dir="./dataset/motions/guoh3dfeats", fps=20.0, normalizer=nomalizer, disable= False, nfeats=263)
sent_emd = SentenceEmbeddings(modelname = "sentence-transformers/all-mpnet-base-v2", path = "./dataset/annotations/humanml3d", preload= True)
cnn_motion_loader = VQMotionDataset(unit_length=2**2)

train_dataset = TextMotionDataset(
    path = './dataset/annotations/humanml3d', 
    split = 'train_half_test_4',
    max_len = args.max_length,
    tokenizer=temollm.model.tokenizer,
    motion_loader = motion_loader,
    cnn_motion_loader=cnn_motion_loader, 
    text_to_sent_emb = sent_emd,
    preload=False,
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
)

temollm.fit(
    train_ds = train_dataset,
    valid_ds = val_dataset,
    threshold = args.threshold,
    batch_size = args.batch_size,
    output_dir = args.save_dir,
    epochs = args.epochs,
    learning_rate = args.lr,
    warmup_steps = args.warmup_steps,
    logging_steps = 50,
    seed=args.seed,
    save_total_limit = 5,
    dataloader_pin_memory=False,
    do_eval=False,
    lr_scheduler_type = "cosine",
    weight_decay=0.1,
)