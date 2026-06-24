
from backend.pg_connection import Connection
from backend.midi_parser import MidiParser
import backend.rnn as rnn

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

import backend.midi_tester as midi_tester

import os

MIDI_DIR = "./backend/midiFiles" # uploaded midi files go here

UPLOAD_DIR = "uploads/midi" # uploaded midi files go here
os.makedirs(UPLOAD_DIR, exist_ok=True)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


dbConnection = Connection( "localhost", "myproject", "matej", "postgres12345" )
dbConnection.createTables()

# api/compose
# api/predict

@app.get("/api/predictor")
def predict():
    return {"message": "I am predictor"}

@app.get("/api/composer")
def compose():
    return {"message": "I am composer"}

@app.post("/api/predictor")
def process(data: dict):
    print("/api/predict: predict data: ")
    print( data)  # see what frontend sent
    return {"result": 1}

@app.post("/api/composer")
def process(data: dict):
    print("/api/compose: compose data: ")
    print( data)  # see what frontend sent
    return {"result": 2}

@app.post("/api/composer/midi")
async def store_midi( midiFile: UploadFile = File(...) ):
    print("/api/compose/midi: upload midi data: ")
    content = await midiFile.read()

    print("Received file:", midiFile.filename)
    print("Size:", len(content))

    filepath = os.path.join(UPLOAD_DIR, midiFile.filename)

    # dbConnection = Connection( "localhost", "myproject", "matej", "postgres12345" )
    #
    # dbConnection.connect()
    #
    # cursor = dbConnection.conn.cursor()
    #
    # cursor.execute("""
    #     INSERT INTO midi_files (filename, path)
    #     VALUES (%s, %s)
    # """, (midiFile.filename, filepath))
    #
    # dbConnection.conn.commit()
    # cursor.close()
    #
    # dbConnection.close()

    # filename = f"{uuid.uuid4()}.mid"
    filename = midiFile.filename
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(content)
        print( "wrote file!!!" )

    print( "    parsing midi file:" )

    parser = MidiParser()
    parser.read_midi( filepath )
    print("    end of parsing midi file !!!!!!!!!!!!")

    songs = []
    songs.append( parser.midi_data )

    generated = rnn.compose( songs ) # todo

    print( "    generated songs:" )
    for note in generated:
        print( note )
        print( " " )

    return {"status": "ok"}

# rest:
# | Method | Meaning |
# | ------ | ------- |
# | GET    | read    |
# | POST   | create  |
# | PUT    | replace |
# | PATCH  | modify  |
# | DELETE | remove  |


#   read midi files to train on



for file in Path( MIDI_DIR ).iterdir():
    parser = MidiParser()

    if file.is_file():
        print("reading fileeeee: ", file.name)

        filepath = os.path.join(MIDI_DIR, file.name)
        parser.read_midi(filepath)

    midi_tester.testMidi( parser.midi_data )
    # midi_tester.playMidi()

    break # todo: delete

    # for item in parser.midi_data:
    #     print(item)




print( "end of reading midi files" )


