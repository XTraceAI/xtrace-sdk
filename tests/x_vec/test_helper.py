


def make_chunk_and_vec(num_chunks: int, embed_size: int) -> dict[str, list]:
    """Helper function to create dummy chunks and vectors for testing."""
    chunks = []
    vectors = []
    for i in range(num_chunks):
        chunk = {
            'chunk_content': f'This is the content of chunk {i}',
            'meta_data': {
                'tag1': 'test data',
                'tag2': str(i)
            }
        }
        vector = [float((i + j) % 2) for j in range(embed_size)]  # simple binary vector
        chunks.append(chunk)

        vectors.append(vector)
    return {'chunks': chunks, 'vectors': vectors}