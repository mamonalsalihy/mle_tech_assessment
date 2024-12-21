import time
import torch
import numpy as np
import pandas as pd
from transformers import T5ForConditionalGeneration, T5TokenizerFast, AutoTokenizer, AutoModel
from tqdm import tqdm 

# Paths to models
t5_model_name = "./t5_fine_tuned_retrieval/checkpoint-600"
retrieval_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Load retrieval model (for chunk selection)
retrieval_tokenizer = AutoTokenizer.from_pretrained(retrieval_model_name)
retrieval_model = AutoModel.from_pretrained(retrieval_model_name).eval().cuda()

# Load T5 model
t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).eval().cuda()

def embed_texts(texts, max_length=128):
    # Batch embedding for a list of texts
    inputs = retrieval_tokenizer(texts, return_tensors='pt', truncation=True, max_length=max_length, padding=True)
    with torch.no_grad():
        outputs = retrieval_model(**{k: v.cuda() for k, v in inputs.items()})
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def retrieve_relevant_chunks_batch(queries, titles, abstracts, chunk_size=150, top_k=3):
    """
    Given lists of queries, titles, and abstracts (each of length N),
    this function:
    1. Splits each abstract into chunks.
    2. Embeds all queries and all chunks in a batched manner.
    3. Computes similarities and retrieves top_k chunks per query.
    Returns a list of lists of top chunks for each input.
    """
    # Step 1: Chunk each abstract
    all_chunks = []
    chunk_offsets = []
    for abstract in abstracts:
        tokens = abstract.split()
        chunks = [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
        chunk_offsets.append(len(chunks))
        all_chunks.extend(chunks)

    # Step 2: Embed all queries and their corresponding titles
    combined_queries = [q + " " + t for q, t in zip(queries, titles)]
    query_embeddings = embed_texts(combined_queries)

    # Step 3: Embed all chunks at once
    chunk_embeddings = embed_texts(all_chunks)

    # Step 4: Compute similarities and select top_k chunks for each query
    # We have query_embeddings of shape (N, D) and chunk_embeddings of shape (M, D)
    # We'll process each query's chunk set individually.
    top_chunks_per_query = []
    start = 0
    for i, q_emb in enumerate(query_embeddings):
        # For query i
        end = start + chunk_offsets[i]
        q_chunk_embs = chunk_embeddings[start:end]
        start = end

        similarities = np.dot(q_chunk_embs, q_emb.T).squeeze()
        top_indices = similarities.argsort()[::-1][:top_k]
        selected = [all_chunks[start - chunk_offsets[i] + idx] for idx in top_indices]
        top_chunks_per_query.append(selected)

    return top_chunks_per_query

def generate_answers_batch(queries, titles, top_chunks_per_query, max_input_length=1024, max_target_length=128):
    """
    Given queries, titles, and their retrieved top chunks, perform a single batched generation.
    """
    # Create the input strings for T5
    input_texts = []
    for query, title, chunks in zip(queries, titles, top_chunks_per_query):
        context = " ".join(chunks)
        input_text = f"question: {query} title: {title} context: {context}"
        input_texts.append(input_text)
    
    # Tokenize batch
    input_encodings = t5_tokenizer(input_texts, return_tensors="pt", truncation=True, max_length=max_input_length, padding=True)
    input_encodings = {k: v.cuda() for k, v in input_encodings.items()}

    # Generate answers in batch
    with torch.no_grad():
        output_ids = t5_model.generate(
            **input_encodings, 
            max_length=max_target_length, 
            num_beams=4, 
            early_stopping=True
        )

    answers = [t5_tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    return answers

# Batch size
BATCH_SIZE = 20

# Load inference data from CSV
inference_data = pd.read_csv("eval_data.csv")  
total_samples = len(inference_data)

start_time = time.time()

# List to store all answers
all_answers = []

# Process in batches
for start_idx in tqdm(range(0, total_samples, BATCH_SIZE), total=total_samples//BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, total_samples)
    batch_data = inference_data.iloc[start_idx:end_idx]
    
    # Extract query, title, and abstract for the current batch
    queries = batch_data["query"].tolist()
    titles = batch_data["title"].tolist()
    abstracts = batch_data["abstract"].tolist()
    
    # Retrieve relevant chunks and generate answers
    top_chunks_per_query = retrieve_relevant_chunks_batch(queries, titles, abstracts, top_k=3)
    batch_answers = generate_answers_batch(queries, titles, top_chunks_per_query)
    
    # Append the batch's answers to the overall list
    all_answers.extend(batch_answers)

end_time = time.time()
total_time = end_time - start_time

# Save the answers back to a CSV file
output_df = inference_data.copy()
output_df["label"] = all_answers
output_df.to_csv("eval_results.csv", index=False)

print(f"Inference completed for {total_samples} samples.")
print(f"Total time taken: {total_time:.2f} seconds")