import psycopg2

db = psycopg2.connect(
  database="postgres",
  host="localhost",
  user="postgres",
  password="",
  port="5432"
)

def log(data):
  row_id = None
  try:
    with db.cursor() as cursor:
      cursor.execute(
        '''
        INSERT INTO logs (timestamp, prediction, label)
        VALUES (%s, %s, %s)
        RETURNING id;
        ''',
        (
          data['timestamp'],
          data['prediction'],
          data['label']
        )
      )
      rows = cursor.fetchone()
      if rows:
        row_id = rows[0]
      db.commit()
  except (Exception, psycopg2.DatabaseError) as error:
    print(error)
  finally:
    return row_id

def fetch_logs(limit=100):
  logs = []
  try:
    with db.cursor() as cursor:
      cursor.execute(
        '''
        SELECT id, timestamp, prediction, label
        FROM logs
        ORDER BY timestamp DESC
        LIMIT %s;
        ''',
        (limit,)
      )
      logs = cursor.fetchall()
    db.commit()
  except (Exception, psycopg2.DatabaseError) as error:
    print(error)
  finally:
    return logs

def create_logs_table():
  try:
    with db.cursor() as cursor:
      cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS logs (
          id SERIAL PRIMARY KEY,
          timestamp TIMESTAMP NOT NULL,
          prediction TEXT NOT NULL,
          label TEXT NOT NULL
        );
        '''
      )
    db.commit()
  except (Exception, psycopg2.DatabaseError) as error:
    print("Error creating table:", error)
