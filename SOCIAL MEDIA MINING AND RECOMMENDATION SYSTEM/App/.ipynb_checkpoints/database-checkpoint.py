import sqlite3 as sq



class Database: 


    def __init__(self, db_file):
        self.conn = sq.connect(db_file)
        self.cursor = self.conn.cursor()

    def create_credential_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_credential (
                email TEXT NOT NULL, 
                password TEXT NOT NULL
            )
        ''')


    def insert_credential(self, email, password):
        self.cursor.execute('''
            INSERT INTO user_credential VALUES (?, ?)
        ''', (email, password))
        self.conn.commit()


    def get_user_credentials(self):
        
        rows = self.cursor.execute('SELECT email, password FROM user_credential')
        datase = rows.fetchall()
        if datase != []:
            email, password =  datase[-1]

            return (email, password)
        else: 
            return None
        

    def create_user_profile_table(self):
       
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profile (
                profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                contact TEXT,
                address TEXT,
                gender TEXT,
                date_of_birth DATE,
                passport BLOB
            )
        ''')
        self.conn.commit()
        print('profile table created.. ')

