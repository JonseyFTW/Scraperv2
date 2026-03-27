"""Quick check for duplicate cards in the database."""
import psycopg2
import config

conn = psycopg2.connect(config.DATABASE_URL)
cur = conn.cursor()

# Check for duplicate product_ids
cur.execute("""
    SELECT product_id, product_name, COUNT(*) as cnt
    FROM cards
    GROUP BY product_id, product_name
    HAVING COUNT(*) > 1
    ORDER BY cnt DESC
    LIMIT 20
""")
rows = cur.fetchall()
if rows:
    print(f"Found {len(rows)} duplicate product_ids:")
    for r in rows:
        print(f"  {r[0]} | {r[1]} | {r[2]}x")
else:
    print("No duplicate product_ids found")

# Total dupes
cur.execute("""
    SELECT COUNT(*) FROM (
        SELECT product_id FROM cards
        GROUP BY product_id HAVING COUNT(*) > 1
    ) t
""")
print(f"\nTotal product_ids with duplicates: {cur.fetchone()[0]}")

# Total rows
cur.execute("SELECT COUNT(*) FROM cards")
print(f"Total card rows: {cur.fetchone()[0]}")

# Check Michael Egnew specifically
cur.execute("""
    SELECT product_id, product_name, image_path, status
    FROM cards
    WHERE product_name LIKE '%Michael Egnew%'
    LIMIT 10
""")
print(f"\nMichael Egnew cards:")
for r in cur.fetchall():
    print(f"  {r[0]} | {r[1]} | {r[3]} | {r[2]}")

cur.close()
conn.close()
