import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import math
from contextlib import nullcontext
import torch
from model_architecture import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
from finetune_dataset import TextDataset
import torch.nn.functional as F
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer

def get_lr(it):
    """
    Calculate learning rate using a cosine decay schedule with linear warmup.
    """
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def train_epoch(epoch):
    """
    Train the model for one epoch.
    """
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X, Y, loss_mask = X.to(device), Y.to(device), loss_mask.to(device)
        lr = get_lr(epoch * iter_per_epoch + step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if ddp:
            model.require_backward_grad_sync = (gradient_accumulation_steps - 1) == 0

        with ctx:
            logits = model(X, Y)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0, reduce=False)
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % log_interval == 0:
            spend_time = time.time() - start_time
            print(
                f'Epoch:[{epoch}/{max_epoch}]({step}/{iter_per_epoch}) loss:{loss.item():.3f} lr:{optimizer.param_groups[-1]["lr"]:.7f} epoch_Time:{spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60} min'
            )

def init_model():
    """
    Initialize a new Transformer model.
    """
    model_args = {
        'dim': dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_kv_heads': n_heads,
        'vocab_size': 64793,
        'multiple_of': multiple_of,
        'max_seq_len': max_seq_len,
        'dropout': dropout,
    }
    print("Initializing a new model")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    return model

if __name__ == "__main__":
    save_dir = '../finetune_out'
    max_epoch = 2
    log_interval = 50
    gradient_accumulation_steps = 1
    batch_size = 8
    
    max_seq_len = 2048
    dim = 2048
    n_layers = 18
    n_heads = 8
    multiple_of = 32
    dropout = 0.0
    bias = False
    
    learning_rate = 2e-5
    weight_decay = 1e-4
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    
    decay_lr = True
    warmup_iters = 1000
    lr_decay_iters = 50000
    min_lr = 1e-6
    
    backend = 'nccl'
    
    device = 'cuda'
    dtype = 'float16'
    compile = False

    # Initialize DDP if needed
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")
    
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    best_val_loss = 1e9
    
    # Load and shuffle data
    df = pd.read_csv('../finetune_data/finetune_data.csv')
    df = df.sample(frac=1.0)
    tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
    train_ds = TextDataset(df, tokenizer, max_length=512)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )

    # Initialize model
    model = init_model()
    model.load_state_dict(torch.load('../pretrained_model/Luban-1B.pth'))
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    iter_per_epoch = len(train_loader)
    
    if compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)
    
    if ddp:
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model
    
    # Training loop
    for epoch in range(max_epoch):
        train_epoch(epoch)
        if ddp:
            if torch.distributed.get_rank() == 0:
                torch.save(raw_model.state_dict(), f'{save_dir}/epoch_{epoch}.pth')
        else:
            torch.save(raw_model.state_dict(), f'{save_dir}/epoch_{epoch}.pth')
    
    if ddp:
        destroy_process_group()
