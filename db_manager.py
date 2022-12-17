import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def create_exercise(conn, task):
    """
    Create a new exercise
    :param conn:
    :param exercise:
    :return:
    """

    sql = ''' INSERT INTO exercises(date,exercise,amount)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, task)
    conn.commit()

    return cur.lastrowid

def main():
    database = r"D:self-edu\exercise-counter\training_results.db"

    # create a database connection
    conn = create_connection(database)
    with conn:
        # create a new exercise
        exercise = ('03/12', 'Bicep-curls', '20')
        exercise_id = create_exercise(conn, exercise)

main()