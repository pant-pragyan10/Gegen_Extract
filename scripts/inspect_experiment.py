import sqlite3
import sys


def inspect(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for row in cur.execute("SELECT id,name,config FROM experiments"):
        print("Experiment:", row)
    conn.close()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "daten/experiments.db"
    inspect(path)
