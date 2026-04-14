"""Quick diagnostic: show pending + in-flight card counts per sport."""
import database as db

db.init_db()
conn = db.get_connection()
cur = conn.cursor()

print("\n=== Pending cards per sport ===")
cur.execute("""
    SELECT s.sport, COUNT(*) AS n
    FROM cards c JOIN sets s ON s.slug = c.set_slug
    WHERE c.status = 'pending'
    GROUP BY s.sport ORDER BY n DESC
""")
for row in cur.fetchall():
    print(f"  {row[0]:<12} {row[1]:>12,}")

print("\n=== In-flight (processing/downloading) per sport ===")
cur.execute("""
    SELECT s.sport, c.status, COUNT(*) AS n
    FROM cards c JOIN sets s ON s.slug = c.set_slug
    WHERE c.status IN ('processing', 'downloading')
    GROUP BY s.sport, c.status ORDER BY n DESC
""")
for row in cur.fetchall():
    print(f"  {row[0]:<12} {row[1]:<14} {row[2]:>12,}")

db.put_connection(conn)
