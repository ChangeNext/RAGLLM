import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ##LLM
    parser.add_argument('--llm_models', default='facebook/opt-125m', type=str, help='LLM Models')
    
    ##hpyer_parameter
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--gradient_accumulation-steps', default=1, type=int, metavar='N',help='number of gradient accumulation steps')
    parser.add_argument('--threshold', default=0.8, type=float, help='filter threshold')
    parser.add_argument('--shared_emb_dim', default=256, type=int, help='embed dim')
    parser.add_argument('--epochs', default=100, type=int, help='train epochs')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--warmup_steps', default=100, type=int, metavar='N', help='Number of steps before decaying lr.')
    parser.add_argument('--max_length', default=32, type=int, help='text_len')
    parser.add_argument('--input_prompt', default=None, type=str, help="Input prompt for the language model, if any.")
    parser.add_argument('--bf16', action='store_true', help='Use bf16 precision')
    ##TrainingArguments

    ## LoRA
    parser.add_argument('--lora_rank', default=16, type=int, help='LoRA RANK')
    parser.add_argument('--lora_alpha', default=32, type=int, help='LoRA alpha')
    parser.add_argument('--lora_dropout', default=0.05, type=float, help = "LoRA dopout")
    parser.add_argument('--lora_bias', default="none", type=str, help = "LoRA default")
        
    ##
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--save_dir', default="./result/train", type=str, help='seed')
    
    return parser.parse_args()