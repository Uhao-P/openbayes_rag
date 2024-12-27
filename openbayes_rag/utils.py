from dataclasses import dataclass
import numpy as np

from hashlib import md5





@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)
    

def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def quantize_embedding(embedding: np.ndarray, bits=8) -> tuple:
    """Quantize embedding to specified bits"""
    # Calculate min/max values for reconstruction
    min_val = embedding.min()
    max_val = embedding.max()

    # Quantize to 0-255 range
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = np.round((embedding - min_val) * scale).astype(np.uint8)

    return quantized, min_val, max_val

async def get_best_cached_response(
    hashing_kv, current_embedding, similarity_threshold=0.95
):
    """Get the cached response with the highest similarity"""
    try:
        # Get all keys
        all_keys = await hashing_kv.all_keys()
        max_similarity = 0
        best_cached_response = None

        # Get cached data one by one
        for key in all_keys:
            cache_data = await hashing_kv.get_by_id(key)
            if cache_data is None or "embedding" not in cache_data:
                continue

            # Convert cached embedding list to ndarray
            cached_quantized = np.frombuffer(
                bytes.fromhex(cache_data["embedding"]), dtype=np.uint8
            ).reshape(cache_data["embedding_shape"])
            cached_embedding = dequantize_embedding(
                cached_quantized,
                cache_data["embedding_min"],
                cache_data["embedding_max"],
            )

            similarity = cosine_similarity(current_embedding, cached_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_cached_response = cache_data["return"]

        if max_similarity > similarity_threshold:
            return best_cached_response
        return None

    except Exception as e:
        logger.warning(f"Error in get_best_cached_response: {e}")
        return None
