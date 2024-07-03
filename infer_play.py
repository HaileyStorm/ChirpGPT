from gpt2_infer import GPT, GPTConfig
import torch, torchaudio
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from speech_tokenizer import SpeechTokenizer
from scipy.io.wavfile import write


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
model = GPT(GPTConfig())

original_state_dict = torch.load('./log/model_15999.pt', map_location=torch.device('cpu'))

# Corrected state dictionary
state_dict = {
    'model': OrderedDict([
        (key.replace('_orig_mod.', ''), value) for key, value in original_state_dict['model'].items()
    ]),
    'config': original_state_dict['config'],
    'step': original_state_dict['step'],
    'val_loss': original_state_dict['val_loss']
}

model.load_state_dict(state_dict['model'])
model.eval()
model.to(device)

unseen = [4097, 547, 426, 2825, 1441, 2209, 1300, 161, 4097, 1646]
seperator = 4097

num_return_sequences = 1
#5s = 897 tokens for 32khz and 473 for 24khz; 7s = 1142 for 32khz and 665 for 24khz
max_length = 1142

tokens = [seperator]
#tokens = unseen
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
sample_rng = torch.Generator(device=device)
#sample_rng.manual_seed(42)

output_tokens = []

# Initialize the progress bar
with tqdm(total=max_length) as pbar:
    # Set initial progress
    pbar.update(xgen.size(1))
    
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)

            pbar.update(1)  # Update by the number of new tokens added
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        output_tokens.append(tokens)

with torch.no_grad():
    del xgen
    torch.cuda.empty_cache()
tokenizer = SpeechTokenizer(device=device)

def find_last_instance_of_seperator(lst, element=4097):
    reversed_list = lst[::-1]
    try:
        reversed_index = reversed_list.index(element)
        return len(lst) - 1 - reversed_index
    except ValueError:
        raise ValueError
#audio_out = tokenizer.decode(np.array([output_tokens[0][:find_last_instance_of_seperator(output_tokens[0]) + 1]]))
audio_out = tokenizer.decode(np.array([output_tokens[0]]))
print(audio_out)

write('test.wav', tokenizer.sample_rate, audio_out.cpu().detach().numpy())
