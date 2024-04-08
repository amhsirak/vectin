<h1 align="center">Vectin.</h1>

Vectin is a simple vector store built from scratch for text embedding and similarity search. 

Supports embedding vector storage with disk persistence.

### Example

```py
# Tokens Size
max_tokens_count = 768

vectin = Vectin(name="short_term_logs", max_tokens=max_tokens_count, persist_data=True, data_storage_path=None)

# Build Corpus
corpus = [
    "Machine learning and artificial intelligence are gaining popularity.",
    "Python is widely used in data science and web development.",
    "Data science involves analyzing large datasets to extract meaningful insights.",
    "Artificial intelligence is transforming various industries.",
    "Data Science is the most popular field of study in 2021.",
]

# Preprocess
corpus = vectin.chunk_sentences_to_max_tokens(sentences=corpus, max_tokens=max_tokens_count)

# Encode
corpus_vectors = vectin.encode(corpus)

# Save
vectin.insert_vectors(corpus_vectors)

# Similarity Search
query = "data science"
result = vectin.similarity_search(query)
print(result)

# Disk Persistence
vectin.save_to_disk()
```

### Output

```py
['Python is widely used in data science and web development.', '0.447214'],
['Data science involves analyzing large datasets to extract meaningful insights.', '0.447214'],
['Data Science is the most popular field of study in 2021.', '0.426401']
```
