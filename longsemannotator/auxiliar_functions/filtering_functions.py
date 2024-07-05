import numpy as np
import faiss

def create_faiss_index(tensors, dimension, metric):
    """Create and populate a FAISS index."""
    index = faiss.index_factory(dimension, "Flat", metric)
    index.add(np.array(tensors).astype(np.float32))
    return index

def search_faiss_index(index, tensors, k):
    """Search in the FAISS index."""
    return index.search(np.array(tensors).astype(np.float32), k)

def add_closest_indexes_to_df(df, indexes, column_name='closest_indexes'):
    """Add closest indexes to DataFrame."""
    df.insert(len(df.columns), column_name, None)
    for idx, _ in enumerate(range(df.shape[0])):
        df.at[idx, column_name] = indexes[idx]
