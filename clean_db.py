import os
import psycopg2
import pickle
from config.config_manager import ConfigManager
from utils.logger import logger

def clean_database():
    # 1. Load config
    config_manager = ConfigManager()
    db_config = config_manager.config['database']
    
    # Load env variables if they exist
    env_vars = {}
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip('"').strip("'")
                    
    host = env_vars.get('DB_HOST') or db_config.get('host', 'localhost')
    port = env_vars.get('DB_PORT') or db_config.get('port', 5432)
    user = env_vars.get('DB_USER') or db_config.get('user')
    password = env_vars.get('DB_PASSWORD') or db_config.get('password')
    dbname = env_vars.get('DB_NAME') or db_config.get('dbname')
    sslmode = env_vars.get('DB_SSLMODE') or db_config.get('sslmode', 'disable')
    
    logger.info(f"Connecting to database to clear tables...")
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            sslmode=sslmode
        )
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Disable triggers/constraints temporarily or cascade truncate
            logger.info("Truncating database tables...")
            cur.execute("TRUNCATE TABLE live_tracks CASCADE;")
            cur.execute("TRUNCATE TABLE visitor_events CASCADE;")
            cur.execute("TRUNCATE TABLE visitors CASCADE;")
            logger.info("Successfully cleared database tables: live_tracks, visitor_events, visitors.")
            
        conn.close()
    except Exception as e:
        logger.error(f"Failed to clear database tables: {e}")
        logger.info("Proceeding with FAISS index file deletion anyway.")

    # 2. Delete FAISS Index files
    faiss_files = ["faiss_index.bin", "uuid_mapping.pkl"]
    for file in faiss_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                logger.info(f"Successfully deleted FAISS file: {file}")
            except Exception as e:
                logger.error(f"Failed to delete FAISS file {file}: {e}")
        else:
            logger.info(f"FAISS file not found (already clean): {file}")
            
    logger.info("Fresh start cleanup complete. You can now start main.py.")

if __name__ == "__main__":
    clean_database()
