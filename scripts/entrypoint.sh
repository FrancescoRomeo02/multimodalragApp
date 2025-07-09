#!/bin/bash
set -e

echo "Starting MultimodalRAG services..."

echo "Starting Streamlit application on port 8501..."
exec streamlit run streamlit_app/Home.py --server.port=8501 --server.address=0.0.0.0
