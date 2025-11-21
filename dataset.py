import torch
import torch.utils.data as Dataset
from Bio import SeqIO
from tokenizer import Tokenizer

class ProteinDataset(Dataset.Dataset):
    def __init__(self, fasta_file, max_length=256):
        self.sequences = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            self.sequences.append(str(record.seq))
        
        self.tokenizer = Tokenizer()
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        if len(seq) > self.max_length - 2:  
            seq = seq[:self.max_length - 2]  #truncate if seqn too long 

        token_ids = self.tokenizer.encode(seq)
        input_ids = torch.tensor(token_ids)
        labels = input_ids.clone()  

        probability_matrix = torch.full(labels.shape, 0.15) # 15% masking
        special_tokens_mask = [1 if token_id in self.tokenizer.special_tokens else 0 for token_id in token_ids]

        probability_matrix = probability_matrix.masked_fill(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(len(self.tokenizer.vocab), labels.shape)
        input_ids[indices_random] = random_tokens[indices_random]
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_id)], dim=0)
            labels = torch.cat([labels, torch.full((pad_len,), -100)], dim=0)

        return input_ids, labels
    
if __name__ == "__main__":
    
    ds = ProteinDataset("toy_dataset.fasta")
    print(f"Dataset size: {len(ds)}")
    
    inp, lab = ds[0]
    print("\n--- Example 0 ---")
    print(f"Input shape: {inp.shape}")
    print(f"Labels shape: {lab.shape}")
    
    # Show first 20 tokens
    print("\nInput IDs (first 20):", inp[:20].tolist())
    print("Labels    (first 20):", lab[:20].tolist())
    
    # Decode to see what the model sees
    print("\nDecoded Input:", ds.tokenizer.decode(inp[:20].tolist()))


