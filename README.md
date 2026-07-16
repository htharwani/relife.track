# Unique Person Counting System

Production-grade Unique Person Counting System optimized for the Raspberry Pi 5, Hailo AI HAT 8L (13 TOPS), and Sony IMX500 Camera.

## Features
- **Object Detection**: YOLO inference via Hailo
- **Tracking**: ByteTrack
- **Face Recognition**: SCRFD + ArcFace MobileFaceNet via Hailo
- **Person Re-Identification**: RepVGG ReID via Hailo (fallback if face is invisible)
- **Vector Search**: FAISS (Cosine Similarity via L2 Normalized Inner Product)
- **Metadata Storage**: PostgreSQL
- **Automatic Model Discovery**: Recursively searches the system for `.hef` files and configures them.

## Setup

1. **Install System Dependencies**
```bash
sudo apt update
sudo apt install postgresql libpq-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libcamera
```

2. **Database Setup**
```bash
sudo -u postgres psql -c "CREATE DATABASE person_counter;"
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'password';"
```

3. **Install Python Packages**
```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

4. **Environment Inspection**
Run the inspection script to automatically log your system details:
```bash
chmod +x inspect_environment.sh
./inspect_environment.sh
```

5. **Run the System**
The system will automatically find your installed `.hef` models and update `config/config.yaml`.
```bash
python3 main.py
```

To run the system without sending data to the PostgreSQL database, use the `--no-db` flag:
```bash
python3 main.py --no-db
```
