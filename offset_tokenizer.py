import torch, torchaudio
from snac import SNAC
import numpy as np


class AudioTokenizer:
    """
    A class for tokenizing and detokenizing audio data using the SNAC model.

    This tokenizer converts audio waveforms into a flattened, hierarchical representation
    and vice versa. It uses a pre-trained SNAC model to encode audio into multiple tensors
    of varying granularity, which are then flattened into a single sequence of integers.

    Input Structure:
        - A list of audio waveforms (torch tensors) with a sample rate of 24/32/44kHz (depending on selected SNAC model).

    Output Structure:
        A flattened sequence of integers with the following structure:
        [4097, A1, B1, B2, C1, C2, C3, C4, D1, D2, D3, D4, D5, D6, D7, D8, A2, B3, B4, C5, C6, C7, C8, D9, D10, D11, D12, D13, D14, D15, D16, ..., A144, B287, B288, ..., 4097]

        Where:
        - 4097 is the separator token (appears at the start and end of each sequence)
        - A, B, C, and D represent values from the four SNAC tensors respectively, with offsets applied:
          - A (tensor 0): 0-4095
          - B (tensor 1): 4099-8194
          - C (tensor 2): 8196-12291
          - D (tensor 3, when present): 12293-16388

    Vocabulary sizes:
        - 24kHz model: 12292 tokens
        - 32kHz and 44kHz models: 16389 tokens

    The flattened structure preserves the hierarchical relationship between the tensors:
    - For each timestep in the coarsest tensor (A):
        - One value from the first (coarsest) tensor (A)
        - Two values from the second tensor (B)
        - Four values from the third tensor (C)
        - Eight values from the fourth (finest) tensor (D) (for 32kHz and 44kHz models)

    This structure allows for efficient encoding and decoding of audio data while
    maintaining the multi-scale representation provided by the SNAC model.

    Methods:
        encode: Converts audio waveforms to the flattened token representation.
        decode: Reconstructs audio waveforms from the flattened token representation.
    """

    def __init__(self, device = 'cpu') -> None:
        #self.model = torch.compile(SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().to(device))
        self.model = torch.compile(SNAC.from_pretrained("hubertsiuzdak/snac_44khz").eval().to(device))
        #self.sample_rate = 32000
        self.sample_rate = 44000
        self.device = device
        self.separator = 4097
        self.offsets = [0, 4099, 8196, 12293]

    def flatten_tensors(self, tensors):
        flattened = []
        for batch in range(tensors[0].size(0)):
            flattened_list = [self.separator]
            for i in range(tensors[0].size()[1]):
                for t, tensor in enumerate(tensors):
                    indices = [i * (2 ** t) + j for j in range(2 ** t)]
                    flattened_list.extend(tensor[batch][indices].add(self.offsets[t]).tolist())
            flattened_list.append(self.separator)
            flattened.append(flattened_list)
        #print(flattened)
        return flattened

    def reconstruct_single_tensors(self, flattened_output):
        flattened_output = flattened_output[1:-1]  # Remove separators
        #print(flattened_output)
        num_tensors = 4 if max(flattened_output) > 12292 else 3
        tensor_lengths = [1, 2, 4, 8][:num_tensors]
        total_length = sum(tensor_lengths)

        reconstructed = [[] for _ in range(num_tensors)]

        for i in range(0, len(flattened_output), total_length):
            chunk = flattened_output[i:i + total_length]
            start = 0
            for t, length in enumerate(tensor_lengths):
                for token in chunk[start:start + length]:
                    reconstructed[t].append(token - self.offsets[t])
                start += length

        return [torch.tensor(tensor).unsqueeze(0) for tensor in reconstructed]

    # expects list of waveforms formatted in 24/32/44khz mono (depending on SNAC model selection)
    def encode(self, waves):

        audio = torch.stack(waves).to(self.device)

        with torch.inference_mode():
            # Each  code is a time step, e.g. if 6 seconds audio is passed in using 32khz model you'll get 64 codes each representing ~93.75ms (3000 samples) of audio
            codes = self.model.encode(audio)
        #print(f"encode model output (`codes`) shape: {[code.shape for code in codes]}")
            
        #print("Number of tensors:", len(codes))
        #mx = 0
        #for i, code in enumerate(codes):
        #    print(f"\tTensor {i} shape: {code.shape}, min: {torch.min(code)}, max: {torch.max(code)}")
        #    mx = max(torch.max(code), mx)
        #print(f"Max value: {mx}")
        
        del audio

        with torch.no_grad():
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
        return np.array(self.flatten_tensors(codes))
    
    # of (1, T)
    def decode(self, tokens):
        # Split tokens at separators and ensure each chunk has separators at both ends
        chunks = []
        current_chunk = []
        for entry in tokens:
            for i, token in enumerate(entry):
                if token == self.separator and i > 0 and i < len(entry) - 1:
                    current_chunk.append(token)
                    chunks.append(current_chunk)
                    current_chunk = [self.separator]
                else:
                    current_chunk.append(token)
            # Exclude entries that are just one or two separators
            if len(current_chunk) > 2:
                chunks.append(current_chunk)

        all_audio = []
        for chunk in chunks:
            raw = self.reconstruct_single_tensors(chunk)
            # Check for negative values
            for j, tensor in enumerate(raw):
                if torch.any(tensor < 0):
                    raise ValueError(
                        f"Negative values found in tensor {j}. This indicates invalid offsets in the input data (model under-trained?).")

            num_tensors = len(raw)

            codes = [tensor.to(self.device) for tensor in raw]
            print("Reconstructed codes:")
            for i, code in enumerate(codes):
                print(f'Tensor {i} shape: {code.shape}')
                print(f'Tensor {i} content: {code[0][:18].tolist()}..{code[0][-18:].tolist()}')
                print(f'Tensor {i} min: {torch.min(code)}, max: {torch.max(code)}')
                print()

            with torch.inference_mode():
                audio_hat = self.model.decode(codes)

            all_audio.append(audio_hat)

            for code in codes:
                del code

            with torch.no_grad():
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()

        # Concatenate all audio chunks
        full_audio = torch.cat(all_audio, dim=-1)

        return full_audio

