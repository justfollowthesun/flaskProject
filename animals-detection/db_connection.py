from config import *
import psycopg2

def cursor_db(dbname =dbname, user = user,
              password=password, host=host ):
    conn = psycopg2.connect(dbname=dbname, user=user,
                            password=password, host=host)
    cursor = conn.cursor()
    return cursor