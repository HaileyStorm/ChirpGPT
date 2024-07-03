import torch, torchaudio
from snac import SNAC
import numpy as np


class SpeechTokenizer():
    def __init__(self, device = 'cpu') -> None:
        self.model = torch.compile(SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().to(device))
        self.sample_rate = 32000
        self.device = device
        self.separator = 4097

    def flatten_tensors(self, tensors):
        """Safely flattens a list of tensors into a flat list of integers."""
        flattened = []
        for batch in range(tensors[0].size(0)):
            flattened_list = []
            if len(tensors) == 3:
                for i in range(tensors[0].size()[1]):
                    flattened_list.append(self.separator)
                    flattened_list.append(tensors[0][batch][i].item())
                    for j in range(2):
                        flattened_list.append(tensors[1][batch][j + i * 2].item())
                        for k in range(2):
                            # print(k,i)
                            flattened_list.append(
                                tensors[2][batch][k + j * 2 + i * 4].item()
                            )

            if len(tensors) == 4:
                for i in range(tensors[0].size()[1]):
                    flattened_list.append(self.separator)
                    flattened_list.append(tensors[0][batch][i].item())
                    for j in range(2):
                        flattened_list.append(tensors[1][batch][j + i * 2].item())
                        for k in range(2):
                            # print(k,i)
                            flattened_list.append(
                                tensors[2][batch][k + j * 2 + i * 4].item()
                            )
                            for l in range(2):
                                flattened_list.append(
                                    tensors[3][batch][l + k * 2 + j * 4 + i * 8].item()
                                )
            flattened_list.append(self.separator)
            flattened.append(flattened_list)

        #print(flattened)
        return flattened


    def reconstruct_single_tensors(self, flattened_output):
        """Reconstructs the list of tensors from the flattened output."""

        def count_elements_between_hashes(lst):
            try:
                # Find the index of the first '#'
                first_index = lst.index(self.separator)
                # Find the index of the second '#' after the first
                second_index = lst.index(self.separator, first_index + 1)
                # Count the elements between the two indices
                return second_index - first_index - 1
            except ValueError:
                # Handle the case where there aren't enough '#' symbols
                return f"List does not contain two '{self.separator}' separators"

        def remove_elements_before_hash(flattened_list):
            try:
                # Find the index of the first '#'
                first_hash_index = flattened_list.index(self.separator)
                # Return the list starting from the first '#'
                return flattened_list[first_hash_index:]
            except ValueError:
                # Handle the case where there is no '#'
                raise Exception

        def list_to_torch_tensor(tensor1):
            # Convert the list to a torch tensor
            tensor = torch.tensor(tensor1)
            # Reshape the tensor to have size (1, n)
            tensor = tensor.unsqueeze(0)
            return tensor
        
        flattened_output = flattened_output.tolist()
        flattened_output = remove_elements_before_hash(flattened_output)
        codes = []
        tensor1 = []
        tensor2 = []
        tensor3 = []
        tensor4 = []

        n_elements = count_elements_between_hashes(flattened_output)
        #print("n_elements:", n_elements)
        if n_elements == 7:
            for i in range(0, len(flattened_output), 8):

                tensor1.append(flattened_output[i + 1])
                tensor2.append(flattened_output[i + 2])
                tensor3.append(flattened_output[i + 3])
                tensor3.append(flattened_output[i + 4])

                tensor2.append(flattened_output[i + 5])
                tensor3.append(flattened_output[i + 6])
                tensor3.append(flattened_output[i + 7])
                codes = [
                    list_to_torch_tensor(tensor1),
                    list_to_torch_tensor(tensor2),
                    list_to_torch_tensor(tensor3),
                ]

        if n_elements == 15:
            for i in range(0, len(flattened_output), 16):
                #print(f"{len(flattened_output)} vs {i}")
                tensor1.append(flattened_output[i + 1])
                tensor2.append(flattened_output[i + 2])
                tensor3.append(flattened_output[i + 3])
                tensor4.append(flattened_output[i + 4])
                tensor4.append(flattened_output[i + 5])
                tensor3.append(flattened_output[i + 6])
                tensor4.append(flattened_output[i + 7])
                tensor4.append(flattened_output[i + 8])

                tensor2.append(flattened_output[i + 9])
                tensor3.append(flattened_output[i + 10])
                tensor4.append(flattened_output[i + 11])
                tensor4.append(flattened_output[i + 12])
                tensor3.append(flattened_output[i + 13])
                tensor4.append(flattened_output[i + 14])
                tensor4.append(flattened_output[i + 15])

                codes = [
                    list_to_torch_tensor(tensor1),
                    list_to_torch_tensor(tensor2),
                    list_to_torch_tensor(tensor3),
                    list_to_torch_tensor(tensor4),
                ]

        return codes

    # expects list of waveforms formatted in 24khz)
    def encode(self, waves):

        audio = torch.stack(waves).to(self.device)

        with torch.inference_mode():
            codes = self.model.encode(audio)
        #print(f"encode model output (`codes`) shape: {[code.shape for code in codes]}")
            
        #print("Number of tensors:", len(codes))
        #mx = 0
        #for i, code in enumerate(codes):
        #    print(f"Tensor {i} shape:", code.shape)
        #    mx = max(torch.max(code), mx)
        #print(f"Max value: {mx}")
        
        del audio

        with torch.no_grad():
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
        return np.array(self.flatten_tensors(codes))
    
    # of (1, T)
    def decode(self, tokens):
        # take -1 to remove the end separator.
        raw = [self.reconstruct_single_tensors(x[:-1]) for x in tokens]
        num_tensors = len(raw[0])
        coarse = torch.cat([raw[i][0] for i in range(len(raw))]).to(self.device)
        fine = torch.cat([raw[i][1] for i in range(len(raw))]).to(self.device)
        finer = torch.cat([raw[i][2] for i in range(len(raw))]).to(self.device)
        if num_tensors == 4:
            finest = torch.cat([raw[i][3] for i in range(len(raw))]).to(self.device)
            codes = [coarse, fine, finer, finest]
        else:
            codes = [coarse, fine, finer]
        with torch.inference_mode():
            audio_hat = self.model.decode(codes)

        del coarse
        del fine
        del finer
        del finest
        del codes

        with torch.no_grad():
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
    
        return audio_hat

