# Semantic Chunking - Complete Guide

## Overview

The **semantic chunking** system was integrated into MultimodalRAG to enhance the quality of document segmentation. Unlike classical chunking, which splits text based on length, semantic chunking analyzes content and creates segments based on semantic coherence.

## Key Features

### Advanced Semantic Chunking

* **Semantic analysis**: Uses embeddings to understand text meaning
* **Configurable thresholds**: Adjustable similarity parameters
* **Smart fallback**: Defaults to classical chunking on failure
* **Metadata preservation**: Retains contextual information

### Multimodal Management

* **Smart association**: Links images and tables to relevant text chunks
* **Enriched metadata**: Adds relations between elements
* **Context retention**: Maintains AI-generated and manual descriptions

### Performance Monitoring

* **Detailed metrics**: Tracks chunking performance and quality
* **Automatic comparisons**: Evaluates semantic vs classical chunking
* **Performance charts**: Visual insights into performance
* **Data export**: CSV export for external analysis

## Configuration

### Semantic Chunking Parameters

Defined in `src/config.py`:

```python
# SEMANTIC CHUNKING SETTINGS
SEMANTIC_CHUNK_SIZE = 1000              # Target chunk size
SEMANTIC_CHUNK_OVERLAP = 200            # Chunk overlap
SEMANTIC_THRESHOLD = 0.75               # Semantic similarity threshold
MIN_CHUNK_SIZE = 100                    # Minimum chunk size
CHUNKING_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
```

### Additional Dependencies

```bash
pip install langchain-text-splitters langchain-experimental tiktoken
pip install matplotlib seaborn pandas  # For monitoring
```

## Usage

```python
from src.utils.semantic_chunker import MultimodalSemanticChunker
from src.config import *

# Initialize
chunker = MultimodalSemanticChunker(
    embedding_model=CHUNKING_EMBEDDING_MODEL,
    chunk_size=SEMANTIC_CHUNK_SIZE,
    chunk_overlap=SEMANTIC_CHUNK_OVERLAP,
    semantic_threshold=SEMANTIC_THRESHOLD,
    min_chunk_size=MIN_CHUNK_SIZE
)

# Text chunking
text_elements = [...]  # Your input text elements
chunks = chunker.chunk_text_elements(text_elements)

# Media association
enriched_texts, enriched_images, enriched_tables = \
    chunker.associate_media_to_chunks(chunks, images, tables)

# Statistics
stats = chunker.get_chunking_stats(enriched_texts)
```

## Integration with Indexer

### Updated DocumentIndexer

```python
from src.pipeline.indexer_service import DocumentIndexer

# Semantic chunking (default)
indexer = DocumentIndexer(embedder, use_semantic_chunking=True)

# Classical chunking
indexer = DocumentIndexer(embedder, use_semantic_chunking=False)
```

### Enriched Metadata

Semantic chunks include:

```python
{
    "text": "...",
    "metadata": {
        "chunk_id": "1_0",               # Unique chunk ID
        "chunk_index": 0,                # Position in document
        "total_chunks": 5,               # Total chunks
        "chunk_type": "semantic",        # Chunking type
        "original_length": 2000,         # Original text length
        "chunk_length": 400,             # Chunk length
        "related_images": 2,             # Related images
        "related_tables": 1              # Related tables
        # ... other original metadata
    }
}
```

## Benefits of Semantic Chunking

### Improved Quality

* **Semantic coherence**: More meaningful segments
* **Context preservation**: Maintains sentence flow
* **Adaptive thresholds**: Content-aware configuration

### More Effective Retrieval

* **Higher relevance**: Chunks aligned with user queries
* **Better context**: Conceptual relationships retained
* **Greater precision**: More accurate results

### Flexibility

* **Configurable**: Parameters can be fine-tuned
* **Robust fallback**: Handles edge cases safely
* **Multimodal-ready**: Supports media integration

## Troubleshooting

### Common Issues

1. **Chunking is too slow**

   * Reduce `SEMANTIC_CHUNK_SIZE`
   * Use a lighter embedding model
   * Increase `SEMANTIC_THRESHOLD`

2. **Too many small chunks**

   * Increase `MIN_CHUNK_SIZE`
   * Lower `SEMANTIC_THRESHOLD`
   * Check input text quality

3. **Frequent fallbacks**

   * Check embedding model connectivity
   * Validate input text formatting
   * Increase timeout/memory allocation

### Logging and Debugging

```python
import logging
logging.getLogger('src.utils.semantic_chunker').setLevel(logging.DEBUG)
```

## Conclusion

Semantic chunking significantly improves the quality of multimodal RAG by offering:

* Smarter, semantically meaningful segmentation
* Better context preservation
* Optimized integration of multimedia elements
* Full monitoring and performance tracking

Semantic chunking is recommended as the default method for most use cases, with classical chunking available as a reliable fallback.
