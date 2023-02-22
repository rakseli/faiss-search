import sqlite3


def create_connection(db_file):
    connection = None
    try:
        connection = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)

    return connection

def create_table(connection, tables):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param tables: a CREATE TABLE statement
    :return:
    """
    try:
        c = connection.cursor()
        c.execute(tables)
    except sqlite3.Error as e:
        print("TABLE NOT CREATED")
        print(e)



def add_document(connection,document):
    """
    Create a new document into the documents table
    :param connection:
    :param document:
    :return: project id
    """
    sql = ''' INSERT INTO documents (document_id)
              VALUES(?);'''
    cur = connection.cursor()
    cur.execute(sql, (document,))
    connection.commit()
    return cur.lastrowid

def add_sentence(connection,data):
    """
    Create a new sentence into the sentences table
    :param connection:
    :param sentences:
    :return: project id
    """
    sql = ''' INSERT INTO sentences(id,content,document_id)
              VALUES(?,?,?) '''
    cur = connection.cursor()
    cur.execute(sql,data)
    connection.commit()
    return cur.lastrowid


def select_sentences_document_id(connection, doc_id):
    """
    Query documents by sentences
    :param conn: the Connection object
    :param doc_id:
    :return:
    """
    cur = connection.cursor()
    cur.execute("SELECT * FROM sentences WHERE document_id=?",(doc_id,))

    rows = cur.fetchall()

    for row in rows:
        print(row)