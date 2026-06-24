
import psycopg


class Connection:
    def __init__( self, host, dbname, user, password, port=5432 ):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port



    def connect(self):
        try:
            self.conn = psycopg.connect(
                host=self.host,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                port=self.port
            )
            print("Connected to database!")
        except Exception as e:
            print("Connection failed:", e)

    def createTables( self ):
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute("""
                CREATE TABLE IF NOT EXISTS midi_files (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        self.conn.commit()
        print("Tables created")
        self.close()


    def close(self):
        if self.conn:
            self.conn.close()
            print("Connection closed")