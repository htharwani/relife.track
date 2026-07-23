import queue
import threading
from database.postgres_client import PostgresClient
from utils.logger import logger

class AsyncPostgresClient:
    """
    Asynchronous wrapper for PostgresClient.
    Runs database write operations in a background thread to prevent blocking
    the main video processing and streaming pipeline.
    """
    def __init__(self, config):
        self.client = PostgresClient(config)
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("AsyncPostgresClient background worker thread started.")

    def _worker(self):
        while True:
            try:
                task = self.queue.get()
                if task is None:
                    break
                func_name, args, kwargs = task
                
                # Check connection before query execution
                if not self.client.conn or self.client.conn.closed:
                    self.client.reconnect()
                    
                if self.client.conn and not self.client.conn.closed:
                    func = getattr(self.client, func_name)
                    func(*args, **kwargs)
                else:
                    logger.warning(f"Database connection unavailable. Skipping async DB task '{func_name}'.")
                    
                self.queue.task_done()
            except Exception as e:
                logger.error(f"Error executing async DB task in background: {e}")

    # Synchronous query (run on main thread since it's only executed once at startup)
    def get_all_visitors(self):
        return self.client.get_all_visitors()

    # Asynchronous query wrappers (adds tasks to queue and returns immediately)
    def insert_visitor(self, visitor_uuid, embedding_type):
        self.queue.put(("insert_visitor", (visitor_uuid, embedding_type), {}))

    def update_visitor_last_seen(self, visitor_uuid):
        self.queue.put(("update_visitor_last_seen", (visitor_uuid,), {}))

    def log_event(self, visitor_uuid, camera_id="camera_1"):
        self.queue.put(("log_event", (visitor_uuid, camera_id), {}))

    def update_live_track(self, track_id, visitor_uuid):
        self.queue.put(("update_live_track", (track_id, visitor_uuid), {}))

    def delete_stale_tracks(self, timeout_seconds=60):
        self.queue.put(("delete_stale_tracks", (timeout_seconds,), {}))

    def upsert_people_count_hourly(self, camera_id, total_in, total_out, peak_occupancy, avg_occupancy):
        self.queue.put(("upsert_people_count_hourly", (camera_id, total_in, total_out, peak_occupancy, avg_occupancy), {}))

    def get_current_hourly_metrics(self, camera_id):
        return self.client.get_current_hourly_metrics(camera_id)

    def get_daily_visitor_uuids(self, camera_id):
        return self.client.get_daily_visitor_uuids(camera_id)

    def stop(self):
        """Stops the background worker thread gracefully."""
        self.queue.put(None)
        self.worker_thread.join()
