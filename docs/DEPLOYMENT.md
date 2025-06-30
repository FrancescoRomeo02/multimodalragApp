# Guida al Deployment

## Opzioni di Deployment

### 1. Deployment Locale (Sviluppo)

```bash
# 1. Clona il repository
git clone <repository-url>
cd multimodalrag

# 2. Setup ambiente
make setup-dev

# 3. Configura variabili d'ambiente
cp .env.example .env
# Modifica .env con le tue chiavi API

# 4. Avvia Qdrant
make qdrant-start

# 5. Avvia l'applicazione
make run
```

### 2. Deployment con Docker Compose (Consigliato)

```bash
# 1. Configura variabili d'ambiente
cp .env.example .env
# Aggiungi GROQ_API_KEY al file .env

# 2. Avvia tutti i servizi
docker-compose up -d

# 3. Verifica stato dei servizi
docker-compose ps

# 4. Visualizza log
docker-compose logs -f multimodal-rag
```

### 3. Deployment Produzione

#### 3.1 Cloud (AWS/GCP/Azure)

**Requisiti:**
- Istanza con almeno 4GB RAM
- Docker e Docker Compose installati
- Accesso a storage per file PDF
- Certificato SSL per HTTPS

**Steps:**
```bash
# 1. Deploy su cloud instance
scp -r . user@server:/app/multimodalrag
ssh user@server

# 2. Configura ambiente produzione
cd /app/multimodalrag
cp .env.example .env
# Configura variabili per produzione

# 3. Deploy con Docker
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 4. Setup reverse proxy (Nginx)
sudo apt install nginx
sudo cp deployment/nginx.conf /etc/nginx/sites-available/multimodalrag
sudo ln -s /etc/nginx/sites-available/multimodalrag /etc/nginx/sites-enabled/
sudo systemctl reload nginx
```

#### 3.2 Kubernetes

```bash
# 1. Applica configurazioni K8s
kubectl apply -f deployment/k8s/

# 2. Verifica deployment
kubectl get pods -n multimodalrag

# 3. Accedi all'applicazione
kubectl port-forward service/multimodal-rag-service 8501:8501
```

## Configurazioni di Produzione

### Variabili d'Ambiente Critiche

```bash
# Obbligatorie
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=http://qdrant:6333

# Opzionali per produzione
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=20
CACHE_TTL_SECONDS=7200
ENABLE_PERFORMANCE_MONITORING=true

# Security
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
STREAMLIT_SERVER_ENABLE_CORS=false
```

### Monitoring e Logging

```bash
# Avvia stack completo con monitoring
docker-compose --profile monitoring up -d

# Accedi a Grafana
# URL: http://localhost:3000
# User: admin
# Password: admin (o GRAFANA_PASSWORD dal .env)

# Prometheus metrics
# URL: http://localhost:9090
```

### Backup e Recovery

```bash
# Backup dati Qdrant
docker-compose exec qdrant tar -czf /tmp/qdrant-backup.tar.gz /qdrant/storage
docker cp multimodal-rag-qdrant:/tmp/qdrant-backup.tar.gz ./backups/

# Backup applicazione
tar -czf multimodalrag-backup-$(date +%Y%m%d).tar.gz \
  --exclude='.git' \
  --exclude='logs/*' \
  --exclude='.venv' \
  .

# Restore Qdrant
docker-compose down
docker volume rm multimodalrag_qdrant_data
docker-compose up -d qdrant
docker cp ./backups/qdrant-backup.tar.gz multimodal-rag-qdrant:/tmp/
docker-compose exec qdrant tar -xzf /tmp/qdrant-backup.tar.gz -C /
```

## Troubleshooting

### Problemi Comuni

1. **Qdrant non raggiungibile**
   ```bash
   # Verifica connessione
   curl http://localhost:6333/health
   
   # Controlla log
   docker-compose logs qdrant
   ```

2. **Out of Memory**
   ```bash
   # Aumenta memoria Docker
   docker-compose down
   # Modifica docker-compose.yml aggiungendo:
   # deploy:
   #   resources:
   #     limits:
   #       memory: 4G
   docker-compose up -d
   ```

3. **Errori API Groq**
   ```bash
   # Verifica API key
   curl -H "Authorization: Bearer $GROQ_API_KEY" \
        https://api.groq.com/openai/v1/models
   ```

### Performance Tuning

```bash
# 1. Ottimizza batch size per embedding
export DEFAULT_BATCH_SIZE=64

# 2. Configura caching Redis
export REDIS_URL=redis://redis:6379

# 3. Aumenta worker Streamlit
streamlit run streamlit_app/Home.py --server.maxUploadSize=200
```

## Security Checklist

- [ ] Variabili d'ambiente configurate correttamente
- [ ] HTTPS configurato con certificato valido  
- [ ] Firewall configurato (solo porte necessarie aperte)
- [ ] Backup automatici configurati
- [ ] Monitoring e alerting attivi
- [ ] Log rotation configurato
- [ ] Rate limiting implementato
- [ ] Input validation attiva
