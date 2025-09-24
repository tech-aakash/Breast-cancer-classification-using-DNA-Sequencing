import os
import glob
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

# ───── Logging Setup ─────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ───── Configuration ─────
MODEL_NAME = "zhihan1996/DNA_bert_6"
KMER = 6
BATCH_SIZE = 16
NUM_WORKERS = 2  # Tweak depending on your system
ROOT_FOLDER = "/home/azureuser/dna_sequencing/clean_backward_noncan"
OUTPUT_DIR = "/home/azureuser/dna_sequencing/final_embeddings/embeddings_backward_noncan_npy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───── Dataset Class ─────
class DNADataset(Dataset):
    def __init__(self, sequences, tokenizer, kmer=KMER):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.kmer = kmer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        kmer_seq = ' '.join([sequence[i:i+self.kmer] for i in range(0, len(sequence)-self.kmer+1)])
        encoding = self.tokenizer(kmer_seq, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# ───── Device Selection ─────
def get_device():
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")

# ───── Embedding Function ─────
def generate_embeddings(df, model, tokenizer, device):
    dataset = DNADataset(df['sequence'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding batches"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)

# ───── Main File Loop ─────
def process_all_files(folder_path, output_dir):
    import re

    def natural_key(string):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', string)]

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

    parquet_files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")), key=natural_key)
    logger.info(f"Found {len(parquet_files)} Parquet files in {folder_path}")

    for file_path in parquet_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        emb_npy_path = os.path.join(output_dir, f"{file_name}_embeddings.npy")
        id_npy_path = os.path.join(output_dir, f"{file_name}_ids.npy")

        if os.path.exists(emb_npy_path):
            logger.info(f"Skipping {file_name} - embeddings already exist.")
            continue

        logger.info(f"Processing file: {file_name}")
        df = pd.read_parquet(file_path)

        if 'sequence' not in df.columns or 'id' not in df.columns:
            logger.warning(f"Skipping {file_name} - required columns missing.")
            continue

        embeddings = generate_embeddings(df, model, tokenizer, device)

        # Save embeddings and IDs
        np.save(emb_npy_path, embeddings)
        np.save(id_npy_path, df['id'].values)
        logger.info(f"Saved: {emb_npy_path}, {id_npy_path}")

if __name__ == "__main__":
    process_all_files(ROOT_FOLDER, OUTPUT_DIR)
