import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ProteinDataset
from model import ProteinBERT
from tokenizer import Tokenizer
import os


BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100
MAX_LEN = 512
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FASTA_FILE = "uniprot_sprot.fasta"
MODEL_SAVE_PATH = "protein_bert.pth"

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    dataset = ProteinDataset(FASTA_FILE, max_length=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    tokenizer = Tokenizer()
    vocab_size = len(tokenizer.vocab)
    
    model = ProteinBERT(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training Loop
    model.train() 
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward Pass
            logits = model(input_ids)
            
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
        
            loss = criterion(logits_flat, labels_flat)
            
            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()     
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()