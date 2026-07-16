#!/bin/bash

# inspect_environment.sh
# Automated inspection script for Raspberry Pi 5 with Hailo AI HAT

REPORT_FILE="environment_report.txt"

echo "============================================" | tee $REPORT_FILE
echo " Raspberry Pi Environment Inspection Report" | tee -a $REPORT_FILE
echo "============================================" | tee -a $REPORT_FILE
echo "Date: $(date)" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[1] Raspberry Pi Model:" | tee -a $REPORT_FILE
cat /sys/firmware/devicetree/base/model 2>/dev/null | tee -a $REPORT_FILE || echo "Unknown" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[2] OS Version:" | tee -a $REPORT_FILE
cat /etc/os-release | grep PRETTY_NAME | cut -d '"' -f 2 | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[3] Python Version:" | tee -a $REPORT_FILE
python3 --version 2>&1 | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[4] Hailo Runtime Version:" | tee -a $REPORT_FILE
hailortcli --version 2>&1 | grep "HailoRT" | tee -a $REPORT_FILE || echo "HailoRT not found" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[5] Connected Hailo Device:" | tee -a $REPORT_FILE
hailortcli scan 2>/dev/null | tee -a $REPORT_FILE || echo "No Hailo device found or hailortcli not installed" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[6] Connected Cameras:" | tee -a $REPORT_FILE
rpicam-hello --list-cameras 2>&1 | tee -a $REPORT_FILE || echo "rpicam-apps not found" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[7] Installed HEF Models:" | tee -a $REPORT_FILE
find / -name "*.hef" 2>/dev/null | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[8] Available GStreamer Plugins:" | tee -a $REPORT_FILE
gst-inspect-1.0 --version 2>&1 | head -n 1 | tee -a $REPORT_FILE || echo "GStreamer not found" | tee -a $REPORT_FILE
gst-inspect-1.0 hailo 2>/dev/null | grep -i plugin | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[9] Picamera2 Installation:" | tee -a $REPORT_FILE
python3 -c "import libcamera; print('Picamera2/libcamera is installed')" 2>/dev/null || echo "Picamera2/libcamera not found" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[10] OpenCV Version:" | tee -a $REPORT_FILE
python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "OpenCV not found in current env" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[11] ONNX Runtime Version:" | tee -a $REPORT_FILE
python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "ONNX Runtime not found" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[12] PostgreSQL Client Availability:" | tee -a $REPORT_FILE
psql --version 2>&1 | tee -a $REPORT_FILE || echo "PostgreSQL client not found" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[13] FAISS Availability:" | tee -a $REPORT_FILE
python3 -c "import faiss; print('FAISS installed, version:', faiss.__version__)" 2>/dev/null || echo "FAISS not found" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[14] Free RAM:" | tee -a $REPORT_FILE
free -h | grep Mem | awk '{print "Total: " $2 ", Free: " $4}' | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[15] CPU Information:" | tee -a $REPORT_FILE
lscpu | grep "Model name" | sed -e 's/^[ \t]*//' | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[16] Disk Space:" | tee -a $REPORT_FILE
df -h / | grep / | awk '{print "Size: " $2 ", Used: " $3 ", Free: " $4}' | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "[17] Virtual Environment Path:" | tee -a $REPORT_FILE
if [ -n "$VIRTUAL_ENV" ]; then
    echo "$VIRTUAL_ENV" | tee -a $REPORT_FILE
else
    echo "Not currently running inside a virtual environment" | tee -a $REPORT_FILE
fi
echo "" | tee -a $REPORT_FILE

echo "[18] Current Project Path:" | tee -a $REPORT_FILE
pwd | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

echo "Inspection complete. Report saved to $REPORT_FILE"
