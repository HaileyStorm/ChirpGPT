import torch
import numpy as np
from snac import SNAC

class SpeechTokenizer():
    def __init__(self, device = 'cpu') -> None:
        self.model = torch.compile(SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().to(device))
        self.sample_rate = 32000
        self.device = device
        self.n_codebooks = len(self.model.quantizer.quantizers)

    def flatten_tensors(self, tensors, separator=4097):
        flattened = []
        for batch in range(tensors[0].size(0)):
            flattened_list = []
            for i in range(tensors[0].size(1)):
                flattened_list.append(separator)
                flattened_list.append(tensors[0][batch][i].item())
                for j in range(1, len(tensors)):
                    start = i * (2 ** (j - 1))
                    end = (i + 1) * (2 ** (j - 1))
                    flattened_list.extend(tensors[j][batch][start:end].tolist())
            flattened_list.append(separator)
            flattened.append(flattened_list)
        return flattened

    def reconstruct_single_tensors(self, flattened_output, separator=4097):
        def remove_elements_before_separator(flattened_list):
            try:
                first_separator_index = flattened_list.index(separator)
                return flattened_list[first_separator_index:]
            except ValueError:
                raise Exception("No separator found in the list")

        flattened_output = flattened_output.tolist()
        flattened_output = remove_elements_before_separator(flattened_output)
    
        tensors = [[] for _ in range(self.n_codebooks)]
        group_size = sum(2**i for i in range(self.n_codebooks - 1)) + 1
    
        for i in range(1, len(flattened_output) - 1, group_size):
            tensors[0].append(flattened_output[i])
            start = i + 1
            for j in range(1, self.n_codebooks):
                end = start + 2**(j-1)
                tensors[j].extend(flattened_output[start:end])
                start = end    

        # Convert to tensors and reshape
        return [torch.tensor(t).view(1, -1) for t in tensors]

    def encode(self, waves):
        audio = torch.stack(waves).to(self.device)
        with torch.inference_mode():
            codes = self.model.encode(audio)
        return np.array(self.flatten_tensors(codes))
    
    def decode(self, tokens):
        raw = [self.reconstruct_single_tensors(x[:-1]) for x in tokens]
    
        # Determine the expected shape for each codebook
        expected_shapes = [
            (1, raw[0][0].size(1)),  # Shape for the first codebook
            *[(1, raw[0][0].size(1) * (2**i)) for i in range(1, self.n_codebooks)]  # Shapes for subsequent codebooks
        ]
    
        # Reshape and pad (or trim) each codebook tensor to match the expected shape
        codes = []
        for i in range(self.n_codebooks):
            codebook = torch.cat([raw[j][i] for j in range(len(raw))]).to(self.device)
            expected_shape = expected_shapes[i]
        
            if codebook.size(1) < expected_shape[1]:
                # Pad if smaller
                codebook = torch.nn.functional.pad(codebook, (0, expected_shape[1] - codebook.size(1)))
            elif codebook.size(1) > expected_shape[1]:
                # Trim if larger
                codebook = codebook[:, :expected_shape[1]]
        
            codes.append(codebook)
    
        with torch.inference_mode():
            audio_hat = self.model.decode(codes)
    
        with torch.no_grad():
            if 'cuda' in self.device:
                torch.cuda.empty_cache()

        return audio_hat
