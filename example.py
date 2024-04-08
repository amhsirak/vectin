from vectin import Vectin


def similarity_search():
    max_tokens_count = 768
    vectin = Vectin(name="short_term_logs",
                    max_tokens=max_tokens_count,
                    persist_data=True,
                    data_storage_path=None
                    )
    corpus = [
        "Machine learning and artificial intelligence are gaining popularity.",
        "Python is widely used in data science and web development.",
        "Data science involves analyzing large datasets to extract meaningful insights.",
        "Artificial intelligence is transforming various industries.",
        "Data Science is the most popular field of study in 2021.",
    ]
    print("\nBuilding corpus...\n")
    corpus = vectin.chunk_sentences_to_max_tokens(
        sentences=corpus, max_tokens=max_tokens_count)
    corpus_vectors = vectin.encode(corpus)
    vectin.insert_vectors(corpus_vectors)

    print("\nSearching for similarity...\n")
    query = "data science"
    result = vectin.similarity_search(query)
    print("Similarity search result:", result)

    print("\nSaving data to disk...\n")
    vectin.save_to_disk()
    print("Data saved successfully.")


if __name__ == "__main__":
    print("\n✨ ✨ ✨ Vectin Inittt ✨ ✨ ✨\n")
    similarity_search()
