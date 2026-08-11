FROM python:3.11-slim

WORKDIR /app

# system deps for scikit-image / segyio wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download F3 data, train, then launch the dashboard
# --server.headless true  : disables the email prompt that blocks non-interactive starts
# --server.address 0.0.0.0: binds to all interfaces so the container port is reachable
CMD ["bash", "-c", \
     "python data/download_f3.py && \
      python -m deepseis.train --config configs/default.yaml && \
      streamlit run app/dashboard.py \
        --server.address=0.0.0.0 \
        --server.port=8501 \
        --server.headless=true \
        --browser.gatherUsageStats=false"]

EXPOSE 8501
