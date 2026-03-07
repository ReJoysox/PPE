FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для PyAV (streamlit-webrtc) и opencv/ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg pkg-config build-essential \
    libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev \
    libswresample-dev libswscale-dev \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["bash", "-lc", "streamlit run app.py --server.address 0.0.0.0 --server.port 7860 --server.enableCORS false --server.enableXsrfProtection false"]
