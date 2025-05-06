from pymongo import MongoClient

def connect_db():
    client = MongoClient('mongodb+srv://jennycpero:UQHG2Sfufk73KHGX@cluster0.rudlgwh.mongodb.net/?retryWrites=true&w'
                         '=majority&appName=Cluster0')
    db = client['ASRSDB']
    coll = db.asrsColl
    return coll
