FROM python:3.13-slim

WORKDIR /workspace

# Install essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source directories
COPY app/ ./app/
COPY streamlit_app/ ./streamlit_app/
COPY scripts/ ./scripts/

# Expose API and frontend ports
EXPOSE 8000
EXPOSE 8501

ENV PYTHONPATH=.

# Default start command (can be overridden by compose services)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
