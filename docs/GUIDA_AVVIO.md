# Quick Start Guide - MultimodalRAG

## Setup

### Minimum Requirements

* **Docker** installed ([Download Docker](https://docs.docker.com/get-docker/))
* **Groq API Key** (Free – [Get yours here](https://console.groq.com/keys))

### Startup Steps

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd multimodalrag
   ```

2. **Configure API Key**

   ```bash
   cp .env.example .env
   # Edit the .env file and replace "your_groq_api_key_here" with your actual key
   ```

3. **Launch the stack**

   ```bash
   docker-compose up -d
   ```

4. **Open the application**

   * Navigate to: [http://localhost:8501](http://localhost:8501)
   * The app will launch automatically

## Feature Testing

### 1. Upload a PDF

* Use the left sidebar to upload a PDF
* The system will automatically process text, images, and tables

### 2. Test Queries

Try the following prompts:

**General Queries:**

* "What is this document about?"
* "Summarize the key points"

**Image-Based Queries:**

* "Show the images present in the document"
* "What do the figures show?"

**Table Queries:**

* "What data is in the tables?"
* "Show me numerical results"

**Multimodal Queries:**

* "Combine insights from text and images"

## Key Features to Observe

### Core Functionality

* **PDF Upload** – Automatic processing pipeline
* **Multimodal Extraction** – Text, images, tables
* **Semantic Retrieval** – Context-aware answers
* **Source References** – Transparent outputs
* **Modern UI** – Clean and intuitive UX

### Advanced Technical Features

* **AI Vision** – Image captioning with BLIP
* **OCR Support** – Text extraction from images
* **Object Detection** – YOLO-based visual tagging
* **Vector Database** – Semantic similarity via Qdrant
* **LLM Integration** – Answer generation with Groq API

## Troubleshooting

### "Qdrant not reachable" Error

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not running, restart services
docker-compose down
docker-compose up -d
```

### "Invalid API Key" Error

```bash
# Check if the API key is correct in your .env file
cat .env | grep GROQ_API_KEY

# The key must start with "gsk_"
```

### Port Already in Use

```bash
# Change port in docker-compose.yml
# Find the line "8501:8501" and update to "8502:8501"
# Then access the app via http://localhost:8502
```

## Metrics and Evaluation

The system includes a comprehensive metrics framework:

1. **Run metrics demo**:

   ```bash
   python metrics/demo_metrics.py
   ```

2. **Use in your code**:

   ```python
   from metrics.main_metrics import track_retrieval, track_generation
   
   # Track retrieval performance
   track_retrieval(query_id, relevant_docs, retrieved_docs, retrieval_time)
   
   # Track generation quality  
   track_generation(query_id, generated_text, reference_text, generation_time, token_count)
   ```

3. **Export results**:

   ```python
   from metrics.main_metrics import global_metrics_collector
   global_metrics_collector.export_metrics("results.json")
   ```

See [metrics/README.md](../metrics/README.md) for detailed documentation.

## Support

If you experience issues:

1. **Check logs**:

   ```bash
   docker-compose logs -f multimodal-rag
   ```

2. **Full reset**:

   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

3. **Alternative local setup**:

   ```bash
   pip install -r requirements.txt
   docker run -d -p 6333:6333 qdrant/qdrant
   streamlit run streamlit_app/Home.py
   ```

---
