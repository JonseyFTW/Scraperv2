"""Reset stuck in-flight cards (processing/downloading) back to pending."""
import database as db

db.init_db()
conn = db.get_connection()
cur = conn.cursor()
cur.execute("""
    UPDATE cards
    SET status = 'pending', worker_id = NULL
    WHERE status IN ('processing', 'downloading')
""")
print(f"Reset {cur.rowcount} in-flight cards back to pending.")
conn.commit()
db.put_connection(conn)
