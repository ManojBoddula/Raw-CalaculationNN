FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
# Add fastapi and uvicorn if they aren't already in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart

# Copy the rest of the application code
COPY . .

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
