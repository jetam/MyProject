
from backend.pg_connection import Connection
from backend.midi_parser import MidiParser
import backend.rnn as rnn

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

import backend.midi_tester as midi_tester

import backend.music_transformer as transformer
import backend.music_transformerT1 as transformerT1
import backend.music_transformerT2 as transformerT2

import os

MIDI_DIR = "./backend/midiFiles/midiFavourites" # uploaded midi files go here
UPLOAD_DIR = "uploads/midi" # uploaded midi files go here
os.makedirs(UPLOAD_DIR, exist_ok=True)


def readMidiFiles():
    songs = []
    parser = MidiParser(transformer.MAX_VELOCITY, transformer.MAX_TIME, transformer.MAX_DURATION)

    for file in Path(MIDI_DIR).iterdir():

        if file.is_file():
            filepath = os.path.join(MIDI_DIR, file.name)
            parser.read_midi(filepath)

            # midi_tester.testMidi( parser.convertedNotes( parser.midi_data ) )
            songs.append(parser.midi_data)

        # break # test delete

    return songs

def composeSongTransformer( songs ):
    parser = MidiParser(transformer.MAX_VELOCITY, transformer.MAX_TIME, transformer.MAX_DURATION)

    model = transformer.MusicTransformer()
    model = transformer.train(model, songs, epochs=2) # todo

    seedSong = songs[0][:50]

    generated = transformer.compose(model, seedSong, length=200)
    generatedNotes = parser.convertedNotes(generated)

    midi_tester.testMidi(generatedNotes, "midiTransformer")

def composeSongTransformerT1( songs ):
    parser = MidiParser(transformer.MAX_VELOCITY, transformer.MAX_TIME, transformer.MAX_DURATION)

    model = transformerT1.MusicTransformerT1()
    model = transformerT1.train(model, songs, epochs=2) # todo

    seedSong = songs[0][:50]

    generated = transformerT1.compose(model, seedSong, length=200)
    generatedNotes = parser.convertedNotes(generated)

    midi_tester.testMidi(generatedNotes, "midiTransformerT1")

def composeSongTransformerT2( songs ):
    parser = MidiParser(transformer.MAX_VELOCITY, transformer.MAX_TIME, transformer.MAX_DURATION)

    model = transformerT2.MusicTransformerT2()
    model = transformerT2.train(model, songs, epochs=1) # todo

    seedSong = songs[0][:50]

    generated = transformerT2.compose(model, seedSong, length=200)
    generatedNotes = parser.convertedNotes(generated)

    midi_tester.testMidi(generatedNotes, "midiTransformerT2")

def composeSongRNN( songs ):

    parser = MidiParser(transformer.MAX_VELOCITY, transformer.MAX_TIME, transformer.MAX_DURATION)

    generated = rnn.composeMusic(songs) # todo: make so functions will have same structure as in transformer
    generatedNotes = parser.convertedNotes(generated) # todo: need this?
    midi_tester.testMidi(generatedNotes, "midiRNN1.mid" )


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


dbConnection = Connection( "localhost", "myproject", "matej", "postgres12345" )
dbConnection.createTables()

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

    # todo: dont save uploaded midi files. save ai models in database!

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

    # songs = readMidiFiles()
    #
    # generated = rnn.compose( songs )
    #
    # print( "    generated songs:" )
    # for note in generated:
    #     print( note )
    #     print( " " )

    return {"status": "ok"}

# rest:
# | Method | Meaning |
# | ------ | ------- |
# | GET    | read    |
# | POST   | create  |
# | PUT    | replace |
# | PATCH  | modify  |
# | DELETE | remove  |


songs = readMidiFiles()

# generated = rnn.compose( songs )
composeSongRNN( songs )
# composeSongTransformer(songs)
# composeSongTransformerT1(songs)
# composeSongTransformerT2(songs)

