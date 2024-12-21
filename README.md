# Consensus Scientific Question Answering

## Overview
This project is focused on building a system for scientific question answering that meets the latency requirements while maintaining high performance. Given the long lengths of abstracts and the necessity of processing a batch of 20 examples within 3 seconds, the system employs:

1. **A lightweight retriever model** to return the top-k most similar chunks of abstracts based on the query and title.
2. **An abstractive question answering model** to generate concise and precise answers from the retrieved chunks.

## Architecture

### Retriever Model
- **Model**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Functionality**: Retrieves the top-k most relevant chunks of the abstract for each query.

### Question Answering Model
- **Model**: [google/t5-small](https://huggingface.co/google/t5-small)
- **Functionality**: Generates abstractive answers using the query, title, and retrieved chunks as input.

## Evaluation Metrics
Evaluation is conducted using a combination of ROUGE metrics and BERTScore to ensure both technical precision and semantic accuracy.

- **ROUGE Metrics**: Measures overlap between generated text and ground-truth phrasing.
  - ROUGE-1: Measures unigrams.
  - ROUGE-2: Measures bigrams.
  - ROUGE-L: Measures longest common subsequence.
  - ROUGE-LSum: ROUGE-L for summaries.

- **BERTScore**: Captures subtle semantic similarities using a transformer-based approach.
  - **Precision**: Fraction of generated tokens that align semantically with the reference.
  - **Recall**: Fraction of reference tokens captured by the generated text.
  - **F1 Score**: Harmonic mean of precision and recall.

## Results
### Evaluation Scores
| Metric             | Score   |
|--------------------|---------|
| **ROUGE-1**       | 72.06   |
| **ROUGE-2**       | 52.05   |
| **ROUGE-L**       | 70.61   |
| **ROUGE-LSum**    | 70.56   |
| **BERTScore Precision** | 92.57 |
| **BERTScore Recall**    | 93.77 |
| **BERTScore F1**        | 93.16 |

### Latency Results
For a batch of 20 examples, the system achieves a total latency of **1.51 seconds**, well within the 3-second constraint.

### Inference Code
The following code snippet demonstrates the inference pipeline:

```python
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
```

## Installation
To install all required dependencies, use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Future Directions
### Latency Improvements
1. **Quantization**: Reduce model size and computation.
2. **GPU Acceleration**: Leverage GPUs for faster inference.
3. **ONNX Runtime Optimizations**: Optimize the model using ONNX.
4. **Model Distillation**: Use a distilled version of the model for lower latency.

### Performance Improvements
1. **Larger Models**: Employ models with higher capacity for complex reasoning.
2. **Increase `top_k` Chunks**: Retrieve more chunks for improved context.
3. **Extended Training**: Train for more epochs to improve accuracy.
4. **Synthetic Samples**: Create diverse samples to enhance model generalization.

---

## References
- [Sentence Transformers](https://huggingface.co/sentence-transformers)
- [T5 Model](https://huggingface.co/google/t5-small)
