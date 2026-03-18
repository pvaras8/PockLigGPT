# Configuration for training on molecules data

out_dir = 'out-very-big-molecules-selfies-2'
eval_interval = 500  # Adjust as needed
eval_iters = 200
log_interval = 50    # Adjust as needed


always_save_checkpoint = False

wandb_log = False
wandb_project = 'molecules-gpt'
wandb_run_name = 'molecules-gpt'

dataset = 'very_big_molecules_selfies_2'  # Name of your dataset directory

# Objetivo de tokens por batch efectivo (2^19 tokens = 524,288 tokens)
target_tokens = 2**19  # 524,288 tokens

# Parámetros de entrada conocidos
batch_size = 32       # Número de secuencias por paso
block_size = 156        # Número de tokens por secuencia
num_gpus = 8         # Número de GPUs, queremos que sea múltiplo de este número

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

learning_rate = 3e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-5
beta2 = 0.99

warmup_iters = 100

device = 'cuda'  # Use 'cuda' if you have a GPU, else 'cpu'
compile = False  # Set to True if you have Triton installed and want to compile the model
