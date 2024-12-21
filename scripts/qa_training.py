import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5TokenizerFast, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModel
)
import numpy as np
import evaluate
from transformers import DataCollatorForSeq2Seq
from transformers import EvalPrediction

# =====================
# Retrieval Utilities
# =====================

def mean_pooling(model_output, attention_mask):
    # Mean pooling for sentence embeddings
    token_embeddings = model_output[0]  # First element: last hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Retriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
    
    def embed_text(self, text):
        # Encode text into embeddings
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = mean_pooling(outputs, inputs['attention_mask'])
        return embeddings.cpu().numpy()

    def retrieve_relevant_chunks(self, query, title, abstract, chunk_size=150, top_k=3):
        abstract_tokens = abstract.split()
        if len(abstract_tokens) <= chunk_size:
            # Abstract fits in one chunk, just return it
            return [abstract]

        # Split into chunks
        chunks = []
        for i in range(0, len(abstract_tokens), chunk_size):
            chunk = " ".join(abstract_tokens[i:i+chunk_size])
            chunks.append(chunk)

        # Compute embeddings for query+title
        q_embedding = self.embed_text(query + " " + title)

        # Compute embeddings for each chunk
        chunk_embeddings = []
        for chunk in chunks:
            c_emb = self.embed_text(chunk)
            chunk_embeddings.append(c_emb)
        chunk_embeddings = np.vstack(chunk_embeddings)

        # Compute similarity (dot product)
        similarities = np.dot(chunk_embeddings, q_embedding.T).squeeze()
        top_indices = similarities.argsort()[::-1][:top_k]

        relevant_chunks = [chunks[i] for i in top_indices]
        return relevant_chunks

# =====================
# Dataset Class
# =====================

class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, retriever, max_input_length=1024, max_target_length=128, top_k=3, chunk_size=150):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.top_k = top_k
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        query = str(row["query"]).strip()
        title = str(row["title"]).strip()
        abstract = str(row["abstract"]).strip()
        label = str(row["label"]).strip()

        # Retrieve top chunks from abstract
        relevant_chunks = self.retriever.retrieve_relevant_chunks(query, title, abstract, chunk_size=self.chunk_size, top_k=self.top_k)
        context = " ".join(relevant_chunks)

        # Construct input for T5
        input_text = f"question: {query} title: {title} context: {context}"
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            label,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }

    
# =====================
# Main Training Script
# =====================


def main():
    # Hyperparameters
    model_name = "t5-small"  
    output_dir = "./t5_fine_tuned_retrieval"
    max_input_length = 1024
    max_target_length = 128
    num_train_epochs = 5
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    learning_rate = 3e-4
    weight_decay = 0.01
    top_k = 3
    chunk_size = 150

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    df = pd.read_csv("../data/training_data.csv")

    # print(df['abstract_length'].sort_values(ascending=False).tolist())

    train_df = df.sample(frac=0.8, random_state=42)  # 80% for training

    eval_df = df.drop(train_df.index)  # Remaining 20% for evaluation


    # Save the evaluation dataset to a new file

    eval_file = "../data/eval_data.csv"

    eval_df.to_csv(eval_file, index=False)


    # Load tokenizer and model
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # Initialize retriever
    retriever = Retriever(device=device)

    # Create dataset
    train_dataset = QADataset(
        train_df,
        tokenizer,
        retriever,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        top_k=top_k,
        chunk_size=chunk_size
    )

    eval_dataset = QADataset(
        eval_df,
        tokenizer,
        retriever,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        top_k=top_k,
        chunk_size=chunk_size
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def compute_metrics(eval_pred: EvalPrediction):
        """
        Computes ROUGE metrics for the evaluation of text generation tasks.
        Args:
            eval_pred: EvalPrediction object containing logits (model outputs) and labels (ground truth).
        Returns:
            Dictionary with computed ROUGE metrics.
        """
        # Load the ROUGE metric
        rouge_metric = evaluate.load("rouge")
        bert_score_metric = evaluate.load("bertscore")
        logits, labels = eval_pred

        logits = logits[0]
        
        # Decode predictions and labels
        predictions = np.argmax(logits, axis=-1)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Align labels and predictions for truncation (optional if there's extra padding)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute the ROUGE metric
        rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        bertscore_result = bert_score_metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

        # Since results are already scalar values, we can directly use them
        rouge_result = {f"rouge-{key}": value * 100 for key, value in rouge_result.items()}

        # Optionally round for better readability
        rouge_result = {k: round(v, 2) for k, v in rouge_result.items()}

        # Process BERTScore results: Calculate mean of precision, recall, and F1 scores
        bertscore_aggregated = {
            f"bertscore-{key}": round(np.mean(value) * 100, 2) for key, value in bertscore_result.items() if key in ["precision", "recall", "f1"]
        }

        # Combine results
        result = {**rouge_result, **bertscore_aggregated}

        return result

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",  # change if you have a validation set
        save_strategy="epoch",
        logging_steps=500,
        save_total_limit=2,
        fp16=True if device == "cuda" else False,
        dataloader_num_workers=0,
        report_to="none",  # no integrated logging to wandb, etc.
        greater_is_better=False,
        metric_for_best_model='eval_loss'
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
