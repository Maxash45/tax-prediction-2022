# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose port
EXPOSE 5000

# Run using gunicorn (production ready)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "mySite:app"]
