# Stage 1: Build the React frontend
FROM node:20-slim AS frontend-build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Build the FastAPI backend
FROM python:3.12-slim
WORKDIR /app

# System dependencies for scientific python and xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install requirements and add explicit prod webserver dependencies
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Copy the rest of the application
COPY src/ src/
COPY models/ models/
COPY data/ data/

# Copy the built frontend static assets from Stage 1
COPY --from=frontend-build /app/dist/ frontend/dist/

# Make sure Python can find 'src'
ENV PYTHONPATH=/app

# Expose port (Cloud Run sets PORT env var)
ENV PORT=8080
EXPOSE ${PORT}

# Run the app. No --reload in production!
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}
