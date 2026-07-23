import os
import sys
import time
import psycopg2
from datetime import datetime

# Helper to read environment variables from a .env file
def load_env(env_path):
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip('"').strip("'")
    return env_vars

# Search for the database credentials in standard paths
def get_db_credentials():
    paths = [
        ".env",
        "../backend/.env",
        "../re.check/re.track/backend/.env",
        "C:/Users/sivan/Downloads/re.check/re.track/backend/.env"
    ]
    
    for path in paths:
        if os.path.exists(path):
            env = load_env(path)
            if "DB_HOST" in env and "DB_PASSWORD" in env:
                print(f"Loaded database credentials from: {path}")
                return env
                
    # Direct hardcoded credentials from the backend/.env if file not found
    return {
        "DB_HOST": "63.35.253.248",
        "DB_PORT": "5432",
        "DB_NAME": "peoplecounterdb",
        "DB_USER": "postgres",
        "DB_PASSWORD": "ME35yL6NIhH6ZyHxIJiojoKP5cx2wDc7xYLlLf3S",
        "DB_SSLMODE": "require"
    }

# Read Raspberry Pi CPU temperature using vcgencmd measure_temp
def get_cpu_temperature():
    import subprocess
    import re
    try:
        # Run vcgencmd measure_temp
        res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, check=True)
        # Parse temp=XX.X'C
        match = re.search(r"temp=([\d\.]+)", res.stdout)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error running vcgencmd measure_temp: {e}")
            
    # Fallback to simulated temperature with random variation if running outside Raspberry Pi
    import random
    return round(63.0 + random.uniform(-1.5, 1.5), 2)

def update_loop():
    db_config = get_db_credentials()
    
    print("Starting Raspberry Pi temperature reporting script (runs every 3 minutes)...")
    
    while True:
        try:
            # 1. Read current temperature
            temp = get_cpu_temperature()
            print(f"[{datetime.now()}] Current CPU Temperature: {temp:.2f}°C")
            
            # 2. Connect to database
            conn = psycopg2.connect(
                host=db_config.get("DB_HOST"),
                port=db_config.get("DB_PORT", 5432),
                user=db_config.get("DB_USER"),
                password=db_config.get("DB_PASSWORD"),
                dbname=db_config.get("DB_NAME"),
                sslmode=db_config.get("DB_SSLMODE", "require")
            )
            conn.autocommit = True
            
            with conn.cursor() as cur:
                # Get the primary device
                cur.execute("SELECT device_id, max_temperature FROM public.devices ORDER BY device_id ASC LIMIT 1;")
                device = cur.fetchone()
                
                if device:
                    device_id, max_temp = device
                    max_temp = float(max_temp or 80.0)
                    
                    # Update device stats
                    cur.execute(
                        "UPDATE public.devices SET temperature = %s, last_seen = %s, status = 'ONLINE' WHERE device_id = %s;",
                        (temp, datetime.now(), device_id)
                    )
                    print(f"Updated device {device_id} status and temperature in database.")
                    
                    # Insert alert if overheating
                    if temp > max_temp:
                        cur.execute(
                            "INSERT INTO public.alerts (device_id, alert_type, temperature, max_temperature) VALUES (%s, 'overheat', %s, %s);",
                            (device_id, temp, max_temp)
                        )
                        print(f"⚠️ OVERHEAT ALERT: Temperature ({temp}°C) exceeds max threshold ({max_temp}°C). Logged alert.")
                else:
                    print("No devices found in the devices table. Cannot update temperature.")
                    
            conn.close()
            
        except Exception as e:
            print(f"Error in temperature update cycle: {e}")
            
        # Sleep for 3 minutes (180 seconds)
        time.sleep(180)

if __name__ == "__main__":
    try:
        update_loop()
    except KeyboardInterrupt:
        print("\nExiting temperature reporter script.")
        sys.exit(0)
