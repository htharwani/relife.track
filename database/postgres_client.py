import psycopg2
from psycopg2.extras import RealDictCursor
from utils.logger import logger
import uuid
from datetime import datetime

class PostgresClient:
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.connect()
        self._initialize_schema()

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.config.get('host'),
                port=self.config.get('port'),
                user=self.config.get('user'),
                password=self.config.get('password'),
                dbname=self.config.get('dbname')
            )
            self.conn.autocommit = True
            logger.info("Successfully connected to PostgreSQL")
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
                    cur.execute(q)
            logger.info("Database schema initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")

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
