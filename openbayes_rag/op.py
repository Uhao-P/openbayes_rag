import tiktoken
from .splitter import DocumentSeparatorSplitter
from .utils import compute_mdhash_id


def get_chunks(new_docs, overlap_token_size=50, max_token_size=500, separators=None):
    inserting_chunks = {}
    chunks = []
    
    splitter = DocumentSeparatorSplitter(
        chunk_size=max_token_size, chunk_overlap=overlap_token_size, separators=separators)

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens_list = ENCODER.encode_batch(docs, num_threads=16)
    
    for index, tokens in enumerate(tokens_list):
        tokens_list = []
        if separators is None:
            tokens_list = splitter.split_tokens(tokens=tokens)
        else:
            tokens_list = splitter.split_tokens_with_separator(tokens=tokens)
        chunk_token = ENCODER.decode_batch(tokens_list)
        for i, chunk in enumerate(chunk_token):
            chunks.append(
                {
                    "tokens_length": len(tokens_list[i]),
                    "tokens": tokens_list[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )
            
    for chunk in chunks:
        key = chunk["content"] + chunk["full_doc_id"] + chunk["chunk_order_index"]
        inserting_chunks.update({compute_mdhash_id(key, prefix="chunk-"): chunk})

    return inserting_chunks



