import psycopg2
from psycopg2.extras import RealDictCursor
from utils.logger import logger
import uuid
import os
from datetime import datetime

def load_env_file(filepath=".env"):
    env_vars = {}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    env_vars[key] = val
    return env_vars

class PostgresClient:
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.connect()
        self._initialize_schema()

    def connect(self):
        env_vars = load_env_file()
        
        host = env_vars.get('DB_HOST') or self.config.get('host')
        port = env_vars.get('DB_PORT') or self.config.get('port')
        user = env_vars.get('DB_USER') or self.config.get('user')
        password = env_vars.get('DB_PASSWORD') or self.config.get('password')
        dbname = env_vars.get('DB_NAME') or self.config.get('dbname')
        sslmode = env_vars.get('DB_SSLMODE') or self.config.get('sslmode', 'disable')
        
        try:
            self.conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname=dbname,
                sslmode=sslmode
            )
            self.conn.autocommit = True
            logger.info(f"Successfully connected to PostgreSQL at {host}:{port}/{dbname} (sslmode={sslmode})")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _initialize_schema(self):
        """Creates the necessary tables if they don't exist."""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS visitors (
                uuid UUID PRIMARY KEY,
                first_seen TIMESTAMP NOT NULL,
                last_seen TIMESTAMP NOT NULL,
                embedding_type VARCHAR(50) NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS visitor_events (
                id SERIAL PRIMARY KEY,
                visitor_uuid UUID REFERENCES visitors(uuid),
                timestamp TIMESTAMP NOT NULL,
                camera_id VARCHAR(100)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS live_tracks (
                track_id INT PRIMARY KEY,
                visitor_uuid UUID REFERENCES visitors(uuid),
                last_updated TIMESTAMP NOT NULL
            );
            """
        ]
        try:
            with self.conn.cursor() as cur:
                for q in queries:
                    try:
                        cur.execute(q)
                    except Exception as e:
                        logger.warning(f"Could not execute table initialization query: {e}. Resetting transaction.")
                        self.conn.rollback()
            logger.info("Database schema check finished.")
        except Exception as e:
            logger.warning(f"Failed to check/initialize database schema: {e}")

    def insert_visitor(self, visitor_uuid, embedding_type):
        now = datetime.now()
        query = """
        INSERT INTO visitors (uuid, first_seen, last_seen, embedding_type)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (uuid) DO UPDATE 
        SET last_seen = EXCLUDED.last_seen;
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (str(visitor_uuid), now, now, embedding_type))

    def update_visitor_last_seen(self, visitor_uuid):
        now = datetime.now()
        query = "UPDATE visitors SET last_seen = %s WHERE uuid = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, (now, str(visitor_uuid)))

    def log_event(self, visitor_uuid, camera_id="camera_1"):
        now = datetime.now()
        query = """
        INSERT INTO visitor_events (visitor_uuid, timestamp, camera_id)
        VALUES (%s, %s, %s);
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (str(visitor_uuid), now, camera_id))

    def update_live_track(self, track_id, visitor_uuid):
        now = datetime.now()
        query = """
        INSERT INTO live_tracks (track_id, visitor_uuid, last_updated)
        VALUES (%s, %s, %s)
        ON CONFLICT (track_id) DO UPDATE 
        SET visitor_uuid = EXCLUDED.visitor_uuid,
            last_updated = EXCLUDED.last_updated;
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (track_id, str(visitor_uuid), now))
            
    def delete_stale_tracks(self, timeout_seconds=60):
        query = """
        DELETE FROM live_tracks 
        WHERE last_updated < NOW() - INTERVAL '%s seconds';
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (timeout_seconds,))

    def upsert_people_count_hourly(self, camera_id, total_in, total_out, peak_occupancy, avg_occupancy):
        now = datetime.now()
        report_date = now.date()
        report_hour = now.hour
        
        # Check if record exists for this camera, date, and hour
        check_query = """
        SELECT id FROM public.people_count_hourly 
        WHERE camera_id = %s AND report_date = %s AND report_hour = %s;
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(check_query, (camera_id, report_date, report_hour))
                row = cur.fetchone()
                
                if row:
                    # Update existing record
                    update_query = """
                    UPDATE public.people_count_hourly
                    SET total_in = %s,
                        total_out = %s,
                        peak_occupancy = GREATEST(peak_occupancy, %s),
                        avg_occupancy = %s,
                        created_at = %s
                    WHERE id = %s;
                    """
                    cur.execute(update_query, (total_in, total_out, peak_occupancy, float(avg_occupancy), now, row[0]))
                else:
                    # Insert new record
                    insert_query = """
                    INSERT INTO public.people_count_hourly (
                        camera_id, report_date, report_hour, total_in, total_out, peak_occupancy, avg_occupancy, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """
                    cur.execute(insert_query, (camera_id, report_date, report_hour, total_in, total_out, peak_occupancy, float(avg_occupancy), now))
        except Exception as e:
            logger.error(f"Failed to upsert hourly metrics: {e}")
