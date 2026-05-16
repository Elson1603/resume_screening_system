FROM python:3.10-slim

# Create a non-root user that Hugging Face requires (User ID 1000)
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    NLTK_DATA=/home/user/app/nltk_data

# Set the working directory
WORKDIR $HOME/app

# Install system dependencies (often needed for ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLP models
RUN python -m spacy download en_core_web_sm
RUN mkdir -p $NLTK_DATA && \
    python -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA'); nltk.download('stopwords', download_dir='$NLTK_DATA'); nltk.download('wordnet', download_dir='$NLTK_DATA')"

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads data models results

# Change ownership of all files to the non-root user
RUN chown -R user:user $HOME/app

# Switch to the non-root user
USER user

# Expose the Hugging Face Space port
EXPOSE 7860
ENV PORT=7860

# Start the Flask app using Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:app"]