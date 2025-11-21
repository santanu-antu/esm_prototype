import os
from Bio import SeqIO
import random

'''
This is to create a smaller toy dataset for testing purposes.
'''

INPUT_FASTA = "uniprot_sprot.fasta" 
OUTPUT_FASTA = "toy_dataset.fasta"
NUM_SEQUENCES = 1000
MAX_LEN = 256 

def create_toy_dataset():
    sequences = []
    
    for record in SeqIO.parse(INPUT_FASTA, "fasta"):
        seq = str(record.seq)
        if 20 < len(seq) <= MAX_LEN: 
            sequences.append(record)
        if len(sequences) >= NUM_SEQUENCES:
            break
    SeqIO.write(sequences, OUTPUT_FASTA, "fasta")

if __name__ == "__main__":
    create_toy_dataset()
