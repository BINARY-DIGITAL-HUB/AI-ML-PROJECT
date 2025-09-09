class Database: 
    # database: None 
    def __init__(self) :
        # self.database
        with sq.connect('exam_database.db') as db: 
            self.database = db



    def create_student_table(self):

        self.database.execute('''
                    CREATE TABLE IF NOT EXISTS student_table (
                        email primary key, 
                        matric_number TEXT NOT NULL
                    ) 
        ''')

    def create_biodata_table(self):

        self.database.execute('''
                    CREATE TABLE IF NOT EXISTS bio_table (
                        dept TEXT NOT NULL,
                        sex TEXT NOT NULL, 
                        age INT NOT NULL,
                        name VARCHAR NOT NULL, 
                        entry_year INT NOT NULL, 
                        matric  TEXT NOT NULL
                    ) 
        ''')

    def create_exam_record_table(self):

        self.database.execute('''
                    CREATE TABLE IF NOT EXISTS bio_table (
                        level_100 TEXT NOT NULL,
                        level_200 VARCHAR NOT NULL, 
                        level_300 VARCHAR NOT NULL,
                        level_400 VARCHAR NOT NULL
                    ) 
        ''')


    def verify_student(self, email, matric):

        query = 'select email, matric_number from student_table' 
        records  = self.database.execute(query).fetchall()
       
        marker = False 
        for re in records: 
            # st.write(re)
            if re[0].lower() == matric.lower() and re[1].lower() == email.lower() : 
                # st.write(f'{email} {matric} {re[0]}{re[1]}')
                marker = True

        return marker 
 
# ==================== initiating database =======

database  = Database()
database.create_student_table()
database.create_biodata_table()
database.create_exam_record_table()
