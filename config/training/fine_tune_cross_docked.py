# Configuration for training on molecules data

out_dir = 'out-models'
init_ckpt_path = 'out-models/ckpt_zink_chembl.pt'  # Path al checkpoint preentrenado
init_from = 'resume'  # 'scratch' or 'resume' or 'gpt2*'

eval_interval = 100              # eval cada 100 iters
eval_iters = 200                 # sube a 300 si ves ruido
log_interval = 20

always_save_checkpoint = False

wandb_log = False
wandb_project = 'molecules-gpt'
wandb_run_name = 'molecules-gpt'

dataset = 'very_big_molecules_selfies_2/cross_docked_jblasser'  # Name of your dataset directory

# Objetivo de tokens por batch efectivo (2^19 tokens = 524,288 tokens)
target_tokens = 2**19  # 524,288 tokens

# Parámetros de entrada conocidos
batch_size = 16       # Número de secuencias por paso
block_size = 156        # Número de tokens por secuencia
num_gpus = 4         # Número de GPUs, queremos que sea múltiplo de este número

# grad_acc global = target_tokens / (batch_size * block_size)
grad_acc_global = target_tokens // (batch_size * block_size)   # ≈105
# redondear al múltiplo de WORLD_SIZE
grad_acc_global = (grad_acc_global // num_gpus) * num_gpus  # -> 104

gradient_accumulation_steps = grad_acc_global
print(f"gradient_accumulation_steps (global) = {gradient_accumulation_steps}")


# Model parameters 
n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.1

learning_rate   = 5e-6
min_lr          = 5e-7
weight_decay    = 0.01
beta2           = 0.95
max_iters       = 1200
lr_decay_iters  = max_iters
warmup_iters    = 120

device = 'cuda'
compile = False