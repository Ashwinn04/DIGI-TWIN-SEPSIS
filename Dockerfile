# Multi-stage Dockerfile for Sepsis Prediction System
# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application build
FROM base as builder

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p outputs/logs outputs/models outputs/figures

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Stage 3: Production image
FROM builder as production

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "api_service.py"]

# Stage 4: Development image
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    mypy

# Override command for development
CMD ["python", "-m", "uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 5: Dashboard-only image
FROM base as dashboard

# Copy only dashboard files
COPY dashboard_real.py integration_real.py ./
COPY explainability_utils/ ./explainability_utils/
COPY models/ ./models/
COPY outputs/ ./outputs/
# Dataset.csv is not in git: mount it at run time (-v "$PWD/Dataset.csv:/app/Dataset.csv")

# Install Streamlit-specific dependencies
RUN pip install --no-cache-dir streamlit

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit dashboard
CMD ["streamlit", "run", "dashboard_real.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 6: Complete system image
FROM builder as complete

# Copy all necessary files
COPY . .

# Create startup script
RUN echo '#!/bin/bash\n\
if [ "$MODE" = "api" ]; then\n\
    echo "Starting API service..."\n\
    python api_service.py\n\
elif [ "$MODE" = "dashboard" ]; then\n\
    echo "Starting dashboard..."\n\
    streamlit run dashboard_real.py --server.port=8501 --server.address=0.0.0.0\n\
else\n\
    echo "Starting both services..."\n\
    python api_service.py &\n\
    streamlit run dashboard_real.py --server.port=8501 --server.address=0.0.0.0\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Expose both ports
EXPOSE 8000 8501

# Use startup script
CMD ["/app/start.sh"]

# Build instructions:
# 
# For API service only:
# docker build --target production -t sepsis-api .
# docker run -p 8000:8000 sepsis-api
#
# For dashboard only:
# docker build --target dashboard -t sepsis-dashboard .
# docker run -p 8501:8501 sepsis-dashboard
#
# For complete system:
# docker build --target complete -t sepsis-complete .
# docker run -p 8000:8000 -p 8501:8501 sepsis-complete
#
# For development:
# docker build --target development -t sepsis-dev .
# docker run -p 8000:8000 sepsis-dev
