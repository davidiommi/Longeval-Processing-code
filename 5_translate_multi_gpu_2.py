"""
This script translates TREC files from French to English using MarianMT on multiple GPUs.
It reads the content from TREC files, processes the content using SpaCy and pySBD for segmentation, and translates the content using MarianMT.
The translated content is then saved in TREC format.

How to Run the Script:

1. Ensure the Required Files and Databases are Available:
   - TREC files: Files containing the content to be translated.
   - MarianMT model: The MarianMT model to use for translation.

2. Set Up the Directory Structure:
   - Place the TREC files in the appropriate subfolders within the input folder.

3. Run the Script:
   - Use the command line to navigate to the directory containing `5_translate_multi_gpu_2.py`.
   - Execute the script with the following command:
     ```sh
     python 5_translate_multi_gpu_2.py <input_folder> <output_folder> --model_name <model_name> --gpu_indices <gpu_indices> --doc_batch_size <doc_batch_size>
     ```
   - Replace `<input_folder>`, `<output_folder>`, `<model_name>`, `<gpu_indices>`, and `<doc_batch_size>` with the appropriate values.

Example Command:
```sh
python 5_translate_multi_gpu_2.py /path/to/input /path/to/output --model_name Helsinki-NLP/opus-mt-fr-en --gpu_indices 0 1 --doc_batch_size 64
```

Script Overview:
"""

from transformers import MarianMTModel, MarianTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from datetime import timedelta
import pysbd
import os
import argparse
from tqdm import tqdm
import logging
import re
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import math

torch.backends.cudnn.benchmark = True

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables for MarianMT model and tokenizer
_model = None
_tokenizer = None
_generate_kwargs = None

# Initialize the pySBD segmenter for French
_segmenter = pysbd.Segmenter(language="fr", clean=True)

class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

def init_marianmt(rank, model_name="Helsinki-NLP/opus-mt-fr-en"):
    """
    Initialize the MarianMT model and tokenizer for GPU processing.
    """
    global _model, _tokenizer, _generate_kwargs
    _tokenizer = MarianTokenizer.from_pretrained(model_name)
    _model = MarianMTModel.from_pretrained(model_name)
    _model.to(rank)
    _model = DDP(_model, device_ids=[rank])
    _model.eval()
    _generate_kwargs = {
        "max_length": 128,
        "num_beams": 2,
        "early_stopping": True 
    }

def collate_fn(batch):
    """
    Collate function to pad sequences to the same length.
    """
    tokenized_inputs = _tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return tokenized_inputs.input_ids, tokenized_inputs.attention_mask

def translate_text_batch(dataloader, model, device, generate_kwargs):
    """
    Translates a batch of texts from French to English using MarianMT.
    """
    translated_texts = []
    for input_ids, attention_mask in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):  # Use mixed precision
                outputs = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs) if isinstance(model, DDP) else model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        translated_texts.extend([_tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
    return translated_texts

def get_dynamic_batch_size(device, max_batch_size=768, memory_factor=0.8):
    """
    Calculate dynamic batch size based on available GPU memory and utilization.
    """
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    available_memory = total_memory - reserved_memory - allocated_memory
    
    # Adjust memory factor based on current workload
    utilization = torch.cuda.utilization(device)
    if utilization > 0.9:
        memory_factor *= 0.9
    elif utilization < 0.5:
        memory_factor *= 1.1

    batch_size = int((available_memory / total_memory) * max_batch_size * memory_factor)
    return max(1, batch_size)

def process_content_translation_batch(contents, segmenter):
    """
    Segments the content and translates each sentence to English in batches.
    """
    global _model, _tokenizer, _generate_kwargs

    # Segment content into sentences
    try:
        all_sentences = [segmenter.segment(content) for content in contents]
    except re.error as e:
        logging.error(f"Regex error during segmentation: {e}")
        return []

    # Flatten the list of sentences
    flat_sentences = [sentence for sentences in all_sentences for sentence in sentences]

    # Determine dynamic batch size based on GPU memory
    device = torch.device(f"cuda:{dist.get_rank()}")
    batch_size = get_dynamic_batch_size(device)

    # Create a PyTorch DataLoader for efficient batching and streaming
    dataset = SentenceDataset(flat_sentences)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True)  # Pin memory

    # Translate sentences in batches
    translated_sentences = translate_text_batch(dataloader, _model, device, _generate_kwargs)

    # Reconstruct the translated content
    translated_contents = []
    idx = 0
    for sentences in all_sentences:
        translated_contents.append("\n".join(translated_sentences[idx:idx + len(sentences)]))
        idx += len(sentences)

    return translated_contents

def process_trec_file(rank, input_file, output_file, doc_batch_size=32):
    """
    Processes a TREC file by translating the content in the <TEXT> section.
    """
    global _segmenter

    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract DOC elements using regular expressions
    documents = re.findall(r'<DOC>.*?</DOC>', content, re.DOTALL)

    # Partition the documents across ranks
    all_docs_count = len(documents)
    docs_per_rank = max(1, math.ceil(all_docs_count / dist.get_world_size()))
    start_idx = rank * docs_per_rank
    end_idx = min(start_idx + docs_per_rank, all_docs_count)
    partitioned_documents = documents[start_idx:end_idx]

    translated_documents = []
    doc_batches = []
    current_doc_batch = []

    for doc in partitioned_documents:
        docno = re.search(r'<DOCNO>(.*?)</DOCNO>', doc).group(1)
        docid = re.search(r'<DOCID>(.*?)</DOCID>', doc).group(1)
        text = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL).group(1).strip()

        header = f"<DOCNO>{docno}</DOCNO>\n<DOCID>{docid}</DOCID>"
        footer = ""
        current_doc_batch.append((header, text, footer))

        if len(current_doc_batch) >= doc_batch_size:
            doc_batches.append(current_doc_batch)
            current_doc_batch = []

    if current_doc_batch:
        doc_batches.append(current_doc_batch)

    for doc_batch in tqdm(doc_batches, desc=f"Translating {os.path.basename(input_file)}", leave=False):
        headers, bodies, footers = zip(*doc_batch)
        translated_bodies = process_content_translation_batch(bodies, _segmenter)

        for header, translated_body, footer in zip(headers, translated_bodies, footers):
            translated_doc = (
                f"<DOC>\n"
                f"{header.strip()}\n"
                f"<TEXT>\n"
                f"{translated_body.strip()}\n"
                f"</TEXT>\n"
                f"</DOC>"
            ).replace("\n\n", "\n").strip()
            translated_documents.append(translated_doc)

    # Gather partial results from all ranks
    gathered_docs = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_docs, translated_documents)
    merged_documents = [d for docs_chunk in gathered_docs for d in docs_chunk]

    if rank == 0:
        save_trec_file(output_file, merged_documents)

def save_trec_file(output_file, trec_documents):
    """
    Saves the TREC formatted documents to the output file.
    """
    logging.info(f"Saving TREC file to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("\n\n".join(doc.strip() for doc in trec_documents) + "\n")
    logging.info(f"TREC file saved: {output_file}")

def process_folder(rank, input_folder, output_folder, doc_batch_size=32):
    """
    Processes all TREC files in the input folder and saves the translated files in the output folder.
    """
    files_to_process = []
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith('.trec') or file_name.endswith('.jsonl.gz'):
                input_file = os.path.join(root, file_name)
                relative_path = os.path.relpath(root, input_folder)
                output_file = os.path.join(output_folder, relative_path, file_name.replace('.jsonl.gz', '.trec'))
                files_to_process.append((input_file, output_file))

    with tqdm(total=len(files_to_process), desc=f"Processing folder: {input_folder}") as pbar:
        for input_file, output_file in files_to_process:
            tqdm.write(f"Processing file: {input_file}")
            process_trec_file(rank, input_file, output_file, doc_batch_size)
            pbar.update(1)

def setup(rank, world_size, gpu_indices):
    if dist.is_initialized():
        return  # Skip if already initialized

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_indices))
    torch.cuda.set_device(rank)
    
    timeout = timedelta(minutes=20)
    dist.init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size, 
        timeout=timeout
    )

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main(rank, world_size, gpu_indices, input_folder, output_folder, model_name, doc_batch_size):
    setup(rank, world_size, gpu_indices)
    init_marianmt(rank, model_name)  # Initialize model and tokenizer once
    process_folder(rank, input_folder, output_folder, doc_batch_size)
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate TREC files using MarianMT on GPU.")
    parser.add_argument('input_folder', type=str, help="The path to the input folder containing TREC files.")
    parser.add_argument('output_folder', type=str, help="The path to the output folder where translated TREC files will be saved.")
    parser.add_argument('--model_name', type=str, default="Helsinki-NLP/opus-mt-fr-en", help="The name of the MarianMT model to use.")
    parser.add_argument('--gpu_indices', type=int, nargs='+', default=[0], help="Indices of the GPUs to use (e.g., 0 1 2).")
    parser.add_argument('--doc_batch_size', type=int, default=64, help="Number of documents to process in a batch.")
    args = parser.parse_args()

    world_size = len(args.gpu_indices)
    mp.spawn(
        main,
        args=(world_size, args.gpu_indices, args.input_folder, args.output_folder, args.model_name, args.doc_batch_size),
        nprocs=world_size,
        join=True
    )