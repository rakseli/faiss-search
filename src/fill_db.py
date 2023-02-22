from src.get_files import get_files
from src.db_helpers import create_connection, create_table, add_document,add_sentence,select_sentences_document_id
from pathlib import Path
import ufal.udpipe

home = str(Path.home())
udpipemodel=ufal.udpipe.Model.load(f"{home}/models/UDPipe/finnish-tdt.udpipe")
tokenizer=ufal.udpipe.Pipeline(udpipemodel,"tokenizer=ranges","none","none","conllu")
files = get_files(f"{home}/data/finnish_text/Eduskunta/2016-kevät",'transcript')

def get_sentences(parsed):
    # gather the text= lines with the sentences
    sents=[line.replace("# text = ","") for line in parsed.split("\n") if line.startswith("# text = ")]
    return sents

def parse_sentence(raw_str):
    return tokenizer.process(raw_str)

#open file

def create_data(db,files):
    '''
    
    '''
    sql_create_documents_table = """CREATE TABLE IF NOT EXISTS documents (
                                        document_id text PRIMARY KEY
                                    ); """
    sql_create_sentences_table = """CREATE TABLE IF NOT EXISTS sentences (
                                    id integer PRIMARY KEY,
                                    content text NOT NULL,
                                    document_id text NOT NULL,
                                    FOREIGN KEY (document_id) REFERENCES documents (document_id)
                                ); """
    create_table(db,sql_create_documents_table)
    create_table(db,sql_create_sentences_table)
    sentence_id = 0
    for f in files:
        document_id = f.split('/')
        document_id = document_id[-1]
        print(f"Starting to add file {document_id}")
        add_document(db,document_id)
        with open(f,'r') as i_f:
            text = i_f.read()
        
        senteces_conlu=parse_sentence(text)
        sentences = get_sentences(senteces_conlu)
        for s in sentences:
            add_sentence(db,(sentence_id,s,document_id))
            sentence_id+=1
    
    db.close()
if __name__ == '__main__':
    db = create_connection(f"{home}/data/finnish_text/Eduskunta/eduskunta.db")
    #create_data(db,files)
    select_sentences_document_id(db,'2016-04-05.transcript')