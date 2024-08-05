import os
import json
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import namedtuple
from typing import Dict, List, Optional, Union, Callable, Any
from src import utils
from src.model.ACTOR import ACTORStyleEncoder
from src.model.attention import JointCrossAttentionBlock
from transformers import (
    Trainer, 
    PreTrainedModel, 
    PretrainedConfig, 
    TrainerCallback, 
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig
from dataset.collate import collate_text_motion
from src.retrieval import retrieval
import src.model.vqvae as vqvae

class InfoNCE_filtering:
    def __init__(self, 
                 temperature=0.7,
                 threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim
        
    def __call__(self, sim_matrix_1, sim_matrix_2, target, sent_emb=None):
        bs, device = len(sim_matrix_1), sim_matrix_1.device
        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            selfsim = sent_emb @ sent_emb.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            
            sim_matrix_1[idx] = -torch.inf
            sim_matrix_2[idx] = -torch.inf
        
        total_loss = (
            F.cross_entropy(sim_matrix_1, target) + F.cross_entropy(sim_matrix_2, target)
        ) / 2
    
        return total_loss, idx
    
class CustomTrainer(Trainer):
    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.contrastive_loss = InfoNCE_filtering(threshold_selfsim=self.threshold)

    def compute_loss(self, model, inputs, retrun_outputs=False):
        """ Compute loss
        :param model: Huggingface model.
        :param inputs: Dict. Model inputs.
        :param return_outputs: bool. Return outputs or not. Default False.
        :return: torch.Tensor. Loss.
        """
        outputs, last_embedding, last_output_logit, motion_ret_embs = model(
            inputs['token'],
            inputs['motion_x_dict'],
            inputs['cnn_motion'],
            inputs['caption_len']
        )
        sent_emb = inputs["sent_emb"]
        logits_per_motion = motion_ret_embs @ last_embedding.t()
        logits_per_text = logits_per_motion.t()
        target = torch.arange(len(logits_per_motion)).cuda()

        loss, index = self.contrastive_loss(logits_per_motion, logits_per_text, target, sent_emb)
        return (loss, outputs) if retrun_outputs else loss

class TeMoLLM_Config(PretrainedConfig):
    def __init__(self,
                 freeze_mm: bool = True,
                 llm_model: str = 'facebook/opt-125m',
                 shared_emb_dim: Optional[int] = 256,
                 text_emb_layers: List[int] = [-1],
                 dataname: str = "t2m",
                 train_mode: str = True,
                 motion_tmr_hidden_size: int = 256,
                 input_prompt: Optional[str] = None,
                 lora_config: Optional[Dict] = None,
                 bf16 : bool = False,
                 **kwconfig):
        super().__init__(**kwconfig)
        self.freeze_mm = freeze_mm
        self.llm_model = llm_model
        self.shared_emb_dim = shared_emb_dim
        self.text_emb_layers = text_emb_layers
        self.motion_tmr_hidden_size = motion_tmr_hidden_size
        self.dataname = dataname
        self.lora_config = lora_config
        self.input_prompt = input_prompt
        self.train_mode = train_mode
        self.bf16 = bf16

class TeMoLLM_Model(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.args = config
        self.llm_model = self.args.llm_model
        self.shared_emb_dim = self.args.shared_emb_dim
        self.torch_type = torch.float32
        self.bf16 = self.args.bf16
        if self.bf16:
            self.torch_type = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm_model)
        self.lm = AutoModelForCausalLM.from_pretrained(self.args.llm_model, torch_dtype=self.torch_type)

        self.input_prompt = self.args.input_prompt

        peft_config = LoraConfig(**self.args.lora_config)
        # self.logger.info(peft_config)
        self.lm = get_peft_model(self.lm, peft_config)
        self.lm.print_trainable_parameters()
        self.lm.resize_token_embeddings(len(self.tokenizer))

        ##### ---- TMR Encoder ----- ######
        self.motion_encoder = ACTORStyleEncoder(nfeats = 263,
                                                vae = True,
                                                latent_dim = 256,
                                                ff_size= 1024,
                                                num_layers= 6,
                                                num_heads= 4,
                                                dropout= 0.1,
                                                activation="gelu",
                                                bf16 = self.bf16)
        
        print("###### ----load TMR motion encoder weight---- #######")
        state_dict = torch.load("./src/model/pretrained_model/motion_encoder.pt")
        self.motion_encoder.load_state_dict(state_dict)

        ##### ---- CNN Encoder ----- ######
        self.cnn_encoder = vqvae.HumanVQVAE(config,
                        nb_code=512,
                        code_dim=512,
                        output_emb_width=512,
                        down_t=2,
                        stride_t=2,
                        width=512,
                        depth=3,
                        dilation_growth_rate=3,
                        activation='relu',
                        norm=None)
        print("###### ----load temos motion encoder weight---- #######")
        file_path = "./src/model/pretrained_model/t2m.pth"
        ckpt = torch.load(file_path, map_location='cpu')
        self.cnn_encoder.load_state_dict(ckpt['net'], strict=True)

        if self.args.freeze_mm:
          print("###### ----Freezing the Motion_Encoder---- ######")
          for param in self.motion_encoder.parameters():
            param.requires_grad = False
          for param in self.cnn_encoder.parameters():
            param.requires_grad = False

        self.ret_text_hidden_fcs = nn.ModuleList([])
        for layer_idx in self.args.text_emb_layers:
              if (layer_idx == -1 or layer_idx == self.lm.config.num_hidden_layers) and ('Qwen' not in self.args.llm_model) and ('Llama' not in self.args.llm_model)and ('gemma' not in self.args.llm_model):
                  print("OPT")
                  in_dim = self.lm.config.word_embed_proj_dim
                  text_fc = [nn.Linear(in_dim, self.args.shared_emb_dim )]
                  self.ret_text_hidden_fcs.append(nn.Sequential(*text_fc))
              elif ('Qwen' in self.args.llm_model) or ('Llama' in self.args.llm_model) or ('gemma' in self.args.llm_model):
                  print(f"{self.llm_model}")
                  in_dim = self.lm.config.hidden_size
                  text_fc = [nn.Linear(in_dim, self.args.shared_emb_dim )]
                  self.ret_text_hidden_fcs.append(nn.Sequential(*text_fc))
              elif layer_idx < self.lm.config.num_hidden_layers:
                  text_fc = [nn.Linear(self.lm.config.hidden_size, self.shared_emb_dim )]
                  self.ret_text_hidden_fcs.append(nn.Sequential(*text_fc))
              else:
                  raise ValueError(f'Embedding of layer {layer_idx} was requested but model only has {config.lm.config.num_hidden_layers} layers.')

        self.motion_hidden_size = self.args.motion_tmr_hidden_size
        self.motion_emb = nn.Linear(self.motion_hidden_size, self.args.shared_emb_dim)              

        self.joint_cross_attn = JointCrossAttentionBlock(
            dim = 256,
            dim_head = 64,
            context_dim = 512
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_motion_embs_cross_attn(self, motion_x_dict, cnn_motion, bf16=False):
        
        outputs = self.motion_encoder(motion_x_dict)
        mask = motion_x_dict["mask"]
        additional_true = torch.ones(mask.size(0), 2, dtype=torch.bool, device=mask.device)
        new_mask = torch.cat((additional_true, mask), dim=1)
        
        outputs_cnn = self.cnn_encoder.cnn_encode(cnn_motion) #torch.Size([bs, 16, 512])
        if self.bf16:
            outputs_cnn = outputs_cnn.to(torch.bfloat16)

        latent_vectors = self.joint_cross_attn(outputs, outputs_cnn, mask = new_mask)
        latent_vectors = latent_vectors[:,0]

        motion_embs = self.motion_emb(latent_vectors)
        motion_embs = torch.reshape(motion_embs, (motion_embs.shape[0], 1, -1))

        return motion_embs

    def forward(
        self,
        tokenized_data,
        motion_x_dict,
        cnn_motion,
        caption_len):
        
        motion_ret_embs = self.get_motion_embs_cross_attn(motion_x_dict, cnn_motion)
        batch_size, mo_seq_len, _ = motion_ret_embs.shape

        input_ids = tokenized_data.input_ids.squeeze(1)
        attention_mask = tokenized_data.attention_mask.squeeze(1)
        last_embedding_idx = caption_len - 1

        if self.input_prompt is not None:
            input_prompt = self.tokenizer(self.input_prompt, return_tensors='pt')
            input_prompt_ids = input_prompt.input_ids.cuda()
            input_prompt_attention_mask = input_prompt.attention_mask.cuda()

            prompt_len = input_prompt.attention_mask[0].sum()

            input_prompt_ids = input_prompt_ids.repeat(batch_size, 1)
            input_prompt_attention_mask = input_prompt_attention_mask.repeat(batch_size, 1)
            
            input_ids = torch.cat((input_prompt_ids,input_ids), dim=1)
            attention_mask = torch.cat((input_prompt_attention_mask, attention_mask), dim=1)
            last_embedding_idx += prompt_len

        output = self.lm(input_ids = input_ids, 
                         attention_mask = attention_mask,
                         output_hidden_states=True)
        
        last_embedding = None
        last_output_logit = None
        hidden_states = []

        if self.args.shared_emb_dim is not None:
            for idx, fc_layer in zip(self.args.text_emb_layers, self.ret_text_hidden_fcs):
                hidden_states.append(fc_layer(output.hidden_states[idx]))  # (N, seq_len, hidden_size)
        else:
            for idx in self.arsg.text_emb_layers:
                hidden_states.append(output.hidden_states[idx])
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        assert motion_ret_embs.shape[1] == 1, motion_ret_embs.shape

        # for idx in self.args.text_emb_layers:
        #         hidden_states.append(output.hidden_states[idx])
        # last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        last_embedding = torch.stack([last_hidden_state[i, last_embedding_idx[i], :] for i in range(batch_size)], axis=0)  # (N, D)
        last_output_logit = torch.stack([output.logits[i, last_embedding_idx[i] - 1, :] for i in range(batch_size)], axis=0)  # (N, D)         

        motion_ret_embs = motion_ret_embs[:,0,:]
        motion_ret_embs = motion_ret_embs / motion_ret_embs.norm(dim=1, keepdim=True)
        last_embedding = last_embedding / last_embedding.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        motion_ret_embs = logit_scale * motion_ret_embs

        return output, last_embedding, last_output_logit, motion_ret_embs

    def eval_forward(self, batch):
        
        motion_x_dict= batch['motion_x_dict']
        cnn_motion= batch['cnn_motion']
        last_embedding_idx = batch['caption_len'] -1

        motion_ret_embs = self.get_motion_embs_cross_attn(motion_x_dict, cnn_motion)
        batch_size, vis_seq_len, _ = motion_ret_embs.shape

        input_ids = batch['token'].input_ids.squeeze(1).cuda()
        attention_mask = batch['token'].attention_mask.squeeze(1).cuda()
        
        if self.input_prompt is not None:
            input_prompt = self.tokenizer(self.input_prompt, return_tensors='pt')
            input_prompt_ids = input_prompt.input_ids.cuda()
            input_prompt_attention_mask = input_prompt.attention_mask.cuda()

            prompt_len = input_prompt.attention_mask[0].sum()

            input_prompt_ids = input_prompt_ids.repeat(batch_size, 1)
            input_prompt_attention_mask = input_prompt_attention_mask.repeat(batch_size, 1)
            
            input_ids = torch.cat((input_prompt_ids,input_ids), dim=1)
            attention_mask = torch.cat((input_prompt_attention_mask, attention_mask), dim=1)
            last_embedding_idx += prompt_len


        output = self.lm(input_ids = input_ids, 
                         attention_mask = attention_mask,
                         output_hidden_states=True)
        
        last_embedding = None
        last_output_logit = None
        hidden_states = []

        if self.args.shared_emb_dim is not None:
            for idx, fc_layer in zip(self.args.text_emb_layers, self.ret_text_hidden_fcs):
                hidden_states.append(fc_layer(output.hidden_states[idx]))  # (N, seq_len, 2048)
        else:
            for idx in self.arsg.text_emb_layers:
                hidden_states.append(output.hidden_states[idx])
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        assert motion_ret_embs.shape[1] == 1, motion_ret_embs.shape
        # for idx in self.args.text_emb_layers:
        #         hidden_states.append(output.hidden_states[idx])
        # last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        last_embedding = torch.stack([last_hidden_state[i, last_embedding_idx[i], :] for i in range(batch_size)], axis=0)  # (N, D)
        last_output_logit = torch.stack([output.logits[i, last_embedding_idx[i] - 1, :] for i in range(batch_size)], axis=0)  # (N, D)         

        motion_ret_embs = motion_ret_embs[:,0,:]
        motion_ret_embs = motion_ret_embs / motion_ret_embs.norm(dim=1, keepdim=True)
        last_embedding = last_embedding / last_embedding.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        motion_ret_embs = logit_scale * motion_ret_embs

        return last_embedding, motion_ret_embs

class TeMoLLM:
    def __init__(self,
                 model_kwargs: Optional[Dict],
                 max_length: int = 32,
                 device: Optional[str] = None,
                 logger = None,
                 **kwargs: Any):
        super().__init__()
        
        self.max_length = max_length
        self.config = model_kwargs
        self.logger = logger
        self.model = TeMoLLM_Model(config = model_kwargs)
        if self.model.bf16:
            self.logger.info("Model load torch.bfloat16")
            self.model = self.model.to(torch.bfloat16)
            self.model.cnn_encoder = self.model.cnn_encoder.float()

        self.device = device
    
    def cuda(self):
        self.model = self.model.to(torch.device(self.device))
        return self
    
    def fit(self, 
            train_ds: Dataset,
            valid_ds: Optional[Dataset] = None,
            batch_size: int = 32,
            output_dir: Optional[str] = None,
            epochs: int = 1,
            learning_rate: float = 1e-5,
            warmup_steps: int = 0,
            logging_steps: int = 10,
            eval_steps: Optional[int] = None,
            save_steps: int = 100,
            save_strategy: str = 'epoch',
            save_total_limit: int = 10,
            gradient_accumulation_steps: int = 1,
            threshold: float = 0.8,
            bf16: bool = False,
            logger = None,
            **argument_kwargs: Any
            ):
         
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        evaluate_callback = EvaluateCallback(self.model, 
                                             valid_ds,
                                             evaluate_fn=retrieval,
                                             save_dir=output_dir,
                                             bf16 = bf16,
                                             logger=logger
                                            )
        callbacks = [evaluate_callback]

        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            bf16 = bf16,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            **argument_kwargs
        )
        self.logger.info(training_args)
        trainer = CustomTrainer(
            model=self.model,
            train_dataset=train_ds,
            data_collator=collate_text_motion,
            callbacks = callbacks,
            args=training_args,
            threshold = threshold
        )
        trainer.train()
        self.model.save_pretrained(output_dir)
    
    def Text_Motion_Retrieval(self, prompt, max_mot_per_ret):
        prompt += self.model.tokenizer.eos_token

        tokenized_data = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.logit_scale.device)
        
        caption_len = tokenized_data.attention_mask[0].sum()
        last_embedding_idx = caption_len - 1
        input_ids = tokenized_data.input_ids.squeeze(1).cuda()
        attention_mask = tokenized_data.attention_mask.squeeze(1).cuda()
        output = self.model.lm(input_ids = input_ids, 
                         attention_mask = attention_mask,
                         output_hidden_states=True)
        
        hidden_states = []
        for idx, fc_layer in zip(self.model.args.text_emb_layers, self.model.ret_text_hidden_fcs):
                hidden_states.append(fc_layer(output.hidden_states[idx]))  # (N, seq_len, 2048)

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        last_embedding = last_hidden_state[0, last_embedding_idx, :]
        last_embedding = last_embedding / last_embedding.norm(dim=0, keepdim=True)
        
        scores = last_embedding @ self.emb_matrix.T
    
        _, top_motion_idx = scores.topk(max_mot_per_ret)
        return_outputs = []
        motion_outputs = []
        for mot_idx in top_motion_idx:
                #Fine the first image that does not error out.
                try:
                    # seen_motion_idx.append(mot_idx)
                    temp = []
                    motion = self.path_array[mot_idx]
                    ked_id = self.key_id[mot_idx]
                    temp.append(motion)
                    temp.append(ked_id)
                    motion_outputs.append(temp)
                    if len(motion_outputs) == max_mot_per_ret:
                        break
                except:
                    pass
                
        return_outputs.append(motion_outputs)
        
        return return_outputs


    @staticmethod
    def from_pretrained(model_kwargs: Optional[Dict],
                        logger = None,
                        **kwargs: Any):
        
        temollm = TeMoLLM(model_kwargs = model_kwargs, logger = logger,
                          **kwargs)
        return temollm
    
    def save_config(self, fpath: str):
        with open(fpath, 'w', encoding='utf-8') as writer:
            json.dump(self.__cfg, writer, ensure_ascii=False, indent=2)



class EvaluateCallback(TrainerCallback):
    """
    Custom TrainerCallback
    This callback will compute corrcoef for each epoch.

    :param model: PreTrainedModel.
    :param valid_ds: Dataset.
    :param evaluate_fn: Callable. It will receive valid_ds as input like `evaluate_fn(valid_ds)`.
    :param save_dir: Optional[str]. specify dir to save model with best results.
    """
    def __init__(self,
                 model,
                 valid_ds: Dataset,
                 evaluate_fn: Callable,
                 save_dir: Optional[str] = None,
                 logger = None,
                 bf16 = False,
                 push_to_hub: bool = False,
                 hub_model_id: Optional[str] = None,
                 hub_private_repo: bool = True):
        
        super().__init__()
        self.model = model
        self.valid_ds = valid_ds
        self.save_dir = save_dir
        self.evaluate_fn = evaluate_fn
        self.push_to_hub = push_to_hub
        self.hub_model_id = hub_model_id
        self.hub_private_repo = hub_private_repo
        self.logger= logger
        self.best_t2m_R1 = 0
        self.best_m2t_R1 = 0
        self.best_only_t2m = 0
        self.best_only_m2t = 0
        self.bf16 = bf16
        self.current = None         
    # def on_epoch_begin(self, args, state, control, **kwargs):
    def on_epoch_end(self, args, state, control, **kwargs):
        metrics = self.evaluate_fn(protocol="normal", dataset=self.valid_ds, threshold=0.95, model=self.model, device="cuda", save_dir = self.save_dir, train_mode = True, batch_size=137, logger=self.logger, nsim_dataset = None,bf16 = self.model.bf16)
        
        current_t2m_R1 = metrics[0]["t2m/R01"]
        current_m2t_R1 = metrics[0]["m2t/R01"]

        if (current_t2m_R1 > self.best_t2m_R1) and (current_m2t_R1 > self.best_m2t_R1):
            self.current = "best"
            self.best_t2m_R1 = current_t2m_R1
            self.best_m2t_R1 = current_m2t_R1
            print(f"New best t2m/R01 : {self.best_t2m_R1} m2t/R01: {self.best_m2t_R1}")
        elif (current_t2m_R1 > self.best_t2m_R1) :
            self.current = "t2m"
            # self.best_only_t2m = current_t2m_R1
            print(f"New best t2m/R01 : {self.best_t2m_R1}")
        elif (current_m2t_R1 > self.best_m2t_R1):
            self.current = "m2t"
            # self.best_only_mt2 = current_m2t_R1
            print(f"New best m2t/R01: {self.best_m2t_R1}")
        else:
            self.current = "not"
        
        if self.save_dir is not None:
            utils.save_checkpoint({
                'state_dict': self.model.state_dict(),
                'best_score': self.best_t2m_R1}, self.current, self.logger,filename=os.path.join(self.save_dir, 'ckpt'))
            print(f'save to {self.save_dir}')



def load_TeMoLLM(model_dir: str, logger) -> TeMoLLM:
    model_args_path = os.path.join(model_dir, 'config.json')
    model_ckpt_path = os.path.join(model_dir, 'ckpt_best.pth.tar')
    print(model_ckpt_path)
    if not os.path.exists(model_args_path):
        raise ValueError(f'model_args.json does not exist in {model_dir}.')
    
    if not os.path.exists(model_ckpt_path):
        raise ValueError(f'ckpt_best.pth.tar does not exist in {model_dir}.')
    
    with open(model_args_path, 'r') as f:
        model_kwargs = json.load(f)
    
    args = namedtuple('args', model_kwargs)(**model_kwargs)

    model_args = TeMoLLM_Config(lora_config=args.lora_config)
    model_args.llm_model = args.llm_model
    model_args.shared_emb_dim = args.shared_emb_dim
    model_args.freeze_mm = args.freeze_mm
    model_args.text_emb_layers = args.text_emb_layers
    model_args.motion_tmr_hidden_size = args.motion_tmr_hidden_size
    model_args.dataname = args.dataname
    model_args.lora_config = args.lora_config
    model_args.bf16 = args.bf16
    model_args.input_prompt = args.input_prompt

    temollm = TeMoLLM.from_pretrained(model_kwargs = model_args, logger=logger,device="cuda")
    checkpoint = torch.load(model_ckpt_path)
    temollm.model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    temollm = temollm.cuda()

    return temollm

import pickle as pkl
def load_TeMoLLM_Retrieval(model_dir: str, logger) -> TeMoLLM:
    model_args_path = os.path.join(model_dir, 'config.json')
    model_ckpt_path = os.path.join(model_dir, 'ckpt_best.pth.tar')
    embs_paths = os.path.join(model_dir, 'amass_embeddings.pkl')

    if not os.path.exists(model_args_path):
        raise ValueError(f'model_args.json does not exist in {model_dir}.')
    if not os.path.exists(model_ckpt_path):
        raise ValueError(f'ckpt_best.pth.tar does not exist in {model_dir}.')
    if not os.path.exists(embs_paths):
        raise ValueError(f'ckpt_best.pth.tar does not exist in {model_dir}.')

    with open(embs_paths, 'rb') as wf:
        train_embs_data = pkl.load(wf)
        path_ = train_embs_data['paths']
        emb_ = train_embs_data['embeddings']
        keyid = train_embs_data['key_id']

    emb_matrix = torch.stack(emb_, axis=0)
    assert len(path_) == emb_matrix.shape[0], (len(path_), emb_matrix.shape[0])    
    assert len(keyid) == emb_matrix.shape[0]

    with open(model_args_path, 'r') as f:
        model_kwargs = json.load(f)
    args = namedtuple('args', model_kwargs)(**model_kwargs)
    model_args = TeMoLLM_Config(lora_config=args.lora_config)
    model_args.llm_model = args.llm_model
    model_args.shared_emb_dim = args.shared_emb_dim
    model_args.freeze_mm = args.freeze_mm
    model_args.text_emb_layers = args.text_emb_layers
    model_args.motion_tmr_hidden_size = args.motion_tmr_hidden_size
    model_args.dataname = args.dataname
    model_args.lora_config = args.lora_config
    temollm = TeMoLLM.from_pretrained(model_kwargs = model_args, logger=logger,device="cuda")
    checkpoint = torch.load(model_ckpt_path)
    temollm.model.load_state_dict(checkpoint['state_dict'], strict=False)
    temollm = temollm.cuda()
    

    logit_scale = temollm.model.logit_scale.exp()
    emb_matrix = emb_matrix.to(logit_scale.device)
    emb_matrix = emb_matrix / emb_matrix.norm(dim=1, keepdim=True)
    emb_matrix = logit_scale * emb_matrix
    temollm.emb_matrix = emb_matrix.to(torch.float32)
    temollm.path_array = path_
    temollm.key_id = keyid
    return temollm