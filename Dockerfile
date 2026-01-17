

FROM python:3.11-slim

# Prevent interactive install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for image processing (minimal)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgtk2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean

# Copy files
WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Streamlit config for deployment
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start command
CMD ["streamlit", "run", "streamlit_app.py"]
