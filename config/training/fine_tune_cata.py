init_from = 'resume'  
out_dir = 'out-molecules-finetune'   # ← nuevo directorio solo para el FT
eval_interval = 10          #  <– val. muy frecuente
eval_iters     = 49         #  ≈ todo el val‑set
log_interval   = 10
always_save_checkpoint = True

# ------------- dataset -------------------
dataset = 'very_big_molecules_selfies'

# ------------- batch & GPUs --------------
batch_size  = 8             # 8 secuencias por GPU
block_size  = 82
num_gpus    = 2             # SOLO 2 GPUs
gradient_accumulation_steps = 2  #  <– no acumular

# ------------- modelo --------------------
n_layer = 24
n_head  = 16
n_embd  = 1024
dropout = 0.20              #  <– más regularización

# ------------- optimizador ---------------
learning_rate = 5e-5        #  LR bajo = micro‑ajuste
weight_decay   = 0.0        #  sin L2
beta1, beta2   = 0.9, 0.99
grad_clip      = 1.0

# ------------- scheduling ----------------
max_iters       = 200       #  techo generoso (~29 épocas en 1 GPU; 100 en 2 GPUs)
lr_decay_iters  = max_iters #  decaimos hasta el final
min_lr          = 1e-6
warmup_iters    = 10

# ------------- otros ---------------------
device  = 'cuda'
compile = False             #  desactiva compile
