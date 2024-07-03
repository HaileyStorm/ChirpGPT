import torch
import torchaudio
from snac import SNAC
import numpy as np

class SpeechTokenizer:
    def __init__(self, device='cpu'):
        self.model = torch.compile(SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().to(device))
        self.sample_rate = 32000
        self.device = device
        self.separator = 4097

    def flatten_code(self, code):
        flattened = [self.separator]
        for tensor in code:
            flattened.extend(tensor.flatten().tolist())
        flattened.append(self.separator)
        return flattened

    def unflatten_code(self, flattened):
        tensor_sizes = [76, 152, 304, 608]
        tensors = []
        start = 0
        for size in tensor_sizes:
            end = start + size
            tensor = torch.tensor(flattened[start:end], dtype=torch.long, device=self.device).unsqueeze(0)
            tensors.append(tensor)
            start = end
        return tensors

    def encode(self, waves):
        encoded_list = []
        for wave in waves:
            audio = wave.unsqueeze(0).to(self.device)
            with torch.inference_mode():
                codes = self.model.encode(audio)
                #print(f"encode model output (`codes`) shape: {[code.shape for code in codes]}")
            #print(f"flat size: {len(self.flatten_code(codes))}")
            encoded_list.append(self.flatten_code(codes))
        #print(f"encode code list length: {len(encoded_list)}")
        return np.array(encoded_list)

    def decode(self, tokens):
        decoded_list = []
        for token in tokens:
            # Remove all separators
            token = [t for t in token if t != self.separator]
            codes = self.unflatten_code(token)
            #print(f"decode model decode input (`codes` after unflatten) shape: {[code.shape for code in codes]}")
            with torch.inference_mode():
                audio_hat = self.model.decode(codes)
            decoded_list.append(audio_hat)
        return torch.cat(decoded_list, dim=1).squeeze(0)  # Remove batch dimension