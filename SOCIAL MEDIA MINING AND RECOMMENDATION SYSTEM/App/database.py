import sqlite3 as sq



class Database: 


    def __init__(self, db_file):
        self.conn = sq.connect(db_file)
        self.cursor = self.conn.cursor()

    def create_credential_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS credentials_table (
            username TEXT PRIMARY KEY NOT NULL,
            password TEXT (255) NOT NULL
            );
        ''')


    def insert_credential(self, username, password):
        self.cursor.execute('''
            INSERT INTO credentials_table VALUES (?, ?)
        ''', (username, password))
        self.conn.commit()


    def get_user_credentials(self):
        
        rows = self.cursor.execute('SELECT username, password FROM credentials_table')
        datase = rows.fetchall()
        if datase != []:
            email, password =  datase[-1]

            return (email, password)
        else: 
            return None



# =============== FACEBOOK PROFILE DETIALS =================

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
                passport BLOB, 
                friend_count INTEGER, 
                work_place TEXT NOT NULL 
            )
        ''')
        self.conn.commit()
        print('profile table created.. ')


# {
#     "friend_count": 2,
#     "id": "100091362595291",
#     "name": "Hamad Alsuwaidi",
#     "work": "Dubai Electricity and Water Authority - DEWA\nShift Manager\nDubai, United Arab Emirates",
#     "places lived": "Abu Dhabi, United Arab Emirates",
#     "mobile": "+971 50 561 1160",
#     "email address": "hamad1076663@gmail.com"
# }
    
    def create_comment_table(self):
       
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS comment_table (
                comment_id INTEGER PRIMARY KEY ,
                comment TEXT NOT NULL,
                reply TEXT NOT NULL,
                comment_score INTEGER,
                reply_score INTEGER,
                total_score INTEGER,
                comment_sent TEXT,
                reply_sent TEXT
            )
        ''')
        self.conn.commit()
        print('comment table created.. ')


    def create_post_table(self):
       
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS post_table (
                post_id INTEGER PRIMARY KEY ,
                post_text TEXT NOT NULL,
                image_text TEXT NOT NULL,
                text_score INTEGER,
                image_score INTEGER,
                total_score INTEGER
            
            )
        ''')
        self.conn.commit()
        print('post table created.. ')



