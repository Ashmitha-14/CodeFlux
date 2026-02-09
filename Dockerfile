FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if any (none for now)
# RUN apt-get update && apt-get install -y ...

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=src/api/app.py
ENV PYTHONPATH=/app

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.api.app:app"]
