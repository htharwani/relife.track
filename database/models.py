# In a more complex setup, you could use SQLAlchemy ORM here.
# For simplicity and performance on the edge, we use raw psycopg2 queries in postgres_client.py.

class Visitor:
    def __init__(self, uuid, first_seen, last_seen, embedding_type):
        self.uuid = uuid
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.embedding_type = embedding_type
