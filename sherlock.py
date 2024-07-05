import torch, torchaudio
from speech_tokenizer import SpeechTokenizer
import numpy as np
from tqdm import tqdm
import itertools
from pathlib import Path
import os
from huggingface_hub import snapshot_download

snapshot_download(repo_id="eastwind/tiny-sherlock-audio", local_dir='./tiny-sherlock-audio', repo_type='dataset')

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

def batch_list(lst, batch_size):
    it = iter(lst)
    return iter(lambda: list(itertools.islice(it, batch_size)), [])

Path('./data').mkdir(parents=True, exist_ok=True)

tokenizer = SpeechTokenizer(device=device)

seconds_per_chunk = 3  #7s @ 32khz = 1216 (excluding final separator), 2s = 384, 3s = 512
batch_size = 1 #Make sure seconds_per_chunk * batch_size is < shortest file duration
print("batch size:", batch_size)

data_path = "./tiny-sherlock-audio"
file_ext = 'mp3'

for audio_path in sorted([x for x in os.listdir(data_path) if file_ext in x]):
    print("processing: ", audio_path)
    waves = []
    waveform, sample_rate = torchaudio.load(f'{data_path}/{audio_path}', backend='soundfile')

    # Resample to if necessary
    if sample_rate != tokenizer.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=tokenizer.sample_rate)
        waveform = resampler(waveform)

    # Convert to mono by averaging the channels if the audio is stereo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    i = 0
    while (i + 1) * seconds_per_chunk * tokenizer.sample_rate < waveform.shape[-1]:
        waves.append(waveform[:, tokenizer.sample_rate * seconds_per_chunk * i: tokenizer.sample_rate * seconds_per_chunk * (i + 1)])
        i+=1
    waves.append(waveform[:, tokenizer.sample_rate * seconds_per_chunk * i:])
    
    batches = list(batch_list(waves, batch_size))

    single_doc = []
    for batch in tqdm(batches[:-1]):
        encoded_batch = tokenizer.encode(batch)
        encoded_batch = encoded_batch.astype(np.int16)
        #print(f"Shape of batch: {np.array(batch).shape} and encoded batch: {np.array(encoded_batch).shape}")
        for x in encoded_batch:
            single_doc.extend(x[:-1])

    if audio_path.split('_')[1] == '01':
        split = 'val'
    else:
        split = 'train'
    np.save(f"./data/sherlock_{split}_{audio_path.split('_')[1]}", single_doc)
