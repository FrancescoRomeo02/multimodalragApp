# Data Directory

This directory contains all data used by the MultimodalRAG project.

## Structure

- **`raw/`**: Original PDFs and documents to process
- **`processed/`**: Processed data and embeddings cache
- **`models/`**: Pre-trained models (YOLO, embedding models)
- **`temp/`**: Temporary processing files

## Notes

- Files in `temp/` are automatically cleaned
- Models in `models/` can be large (excluded from git if necessary)
- `raw/` contains user input documents
