FROM python:3.10-slim

WORKDIR /app

# System dependencies not required for opencv-python-headless

# Install python dependencies
COPY requirements.txt .
# Add fastapi and uvicorn if they aren't already in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart

# Copy the rest of the application code
COPY . .

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
