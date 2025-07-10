import mysql.connector
def get_db_connection():
return mysql.connector.connect(
host="localhost",
user="root",
password="", # default xampp password is empty
database="hybrid_db"
)