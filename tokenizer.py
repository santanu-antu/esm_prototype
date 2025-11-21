class Tokenizer:
    def __init__(self):
        
        self.special_tokens = ["<pad>", "<mask>", "<cls>", "<eos>", "<unk>"] # <pad> for making all sequences same length, <mask> to hide amino acid, <cls> for start of sequence, <eos> for end of sequence, <unk> for unknown amino acid
    
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY" 
        
        # Build Vocabulary
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        for i, aa in enumerate(self.amino_acids):
            self.vocab[aa] = len(self.special_tokens) + i
            
        # Inverse mapping for decoding
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        self.pad_id = self.vocab["<pad>"]
        self.mask_id = self.vocab["<mask>"]
        self.cls_id = self.vocab["<cls>"]
        self.eos_id = self.vocab["<eos>"]
        self.unk_id = self.vocab["<unk>"]

    def encode(self, sequence):   
        """Converts a protein sequence string to a list of integers."""
        ids = [self.cls_id]
        for char in sequence:
            char = char.upper()
            if char in self.vocab:
                ids.append(self.vocab[char])
            else:
                ids.append(self.unk_id)
        ids.append(self.eos_id)
        return ids

    def decode(self, token_ids):  
        """Converts a list of integers back to a string."""
        tokens = []
        for idx in token_ids:
            if idx in self.idx_to_token:
                tokens.append(self.idx_to_token[idx])
            else:
                tokens.append("<unk>")
        return " ".join(tokens)

if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = Tokenizer()
    print(f"Vocabulary: {tokenizer.vocab}")

    test_seq = "MAK"
    encoded = tokenizer.encode(test_seq)
    print(f"\nTest Sequence: {test_seq}")
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")