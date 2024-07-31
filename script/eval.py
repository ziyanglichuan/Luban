from contextlib import nullcontext
import torch
from model_architecture.py import ModelArgs, Transformer
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer

# Configuration settings
num_samples = 1
max_new_tokens = 128
temperature = 1.0
top_k = 30
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = "float32"
compile = False

# Model parameters
max_seq_len = 2048
dim = 2048
n_layers = 18
n_heads = 8
multiple_of = 32
dropout = 0.0

# Create model arguments dictionary
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

# Set random seeds for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Determine device type and context for AMP
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast()

# Initialize model from checkpoint
ckpt_path = '../pretrained_model/Luban-1B.pth'
state_dict = torch.load(ckpt_path, map_location=device)
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)

# Remove unwanted prefix from state_dict keys
unwanted_prefix = '_orig_mod.'
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        new_key = k[len(unwanted_prefix):]
        state_dict[new_key] = state_dict.pop(k)

model.load_state_dict(state_dict, strict=False)
model.eval()
model.to(device)

# Optionally compile the model for faster inference (requires PyTorch 2.0)
if compile:
    print("Compiling the model...")
    model = torch.compile(model)

# Print total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters is: {total_params}')

# Load the tokenizer
tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')

# Sample data for generation
data = [
    {"question": "用挤出机挤塑生产塑料制品是"},
    {"question": "介绍一下数控加工："},
]

# Generate answers
ans_lst = []
for p in data:
    prompt = p['question']
    
    # Encode prompt and prepare input tensor
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        with ctx:
            # Generate response
            generated_ids = model.generate(input_tensor, num_samples, max_new_tokens, temperature=temperature, top_k=top_k)
            answer = tokenizer.decode(generated_ids[0].tolist()).replace(prompt, '')
            ans_lst.append(answer)
            
            # Print prompt and answer
            print('[prompt]:', prompt)
            print('[answer]:', answer)
            print('---------------')
