
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from music import router as musicRouter
# from market import router as marketRouter # todo

app = FastAPI(title="AI Music + Market Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("start server")

app.include_router(musicRouter.router, prefix="/api/music")

# app.include_router(router, prefix="/api/market") # todo


@app.get("/health")
def health_check():
    return {"status": "ok"}








# from backend.pg_connection import Connection
# import backend.midi_parser as Parser
# import backend.rnn as rnn
#
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
#
# import backend.music_transformer as transformer
# import backend.music_transformerT1 as transformerT1
# import backend.music_transformerT2 as transformerT2
#
# from pydantic import BaseModel
#
# import io
#
# # from api.routes import midi, music, market, models
#
# from backend.music.routes import router as music_router
# from backend.market.routes import router as market_router
#
# MIDI_DIR = "./backend/midiFiles/midiFavourites" # uploaded midi files go here
# # UPLOAD_DIR = "uploads/midi" # uploaded midi files go here
#
#
#
# app = FastAPI()
#
# app.include_router(music_router, prefix="/api/music")
# app.include_router(market_router, prefix="/api/market")
#
#
# class ModelSelect(BaseModel):
#     model_name: str
#
# songs = Parser.readMidiFiles(MIDI_DIR)
#
# transformer.composeSong(songs)
# transformerT1.composeSong(songs)
# transformerT2.composeSong(songs)
# rnn.composeSong(songs)
#
#
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# dbConnection = Connection( "localhost", "myproject", "matej", "postgres12345" )
# dbConnection.createTables()
#
# @app.get("/api/predictor")
# def predict():
#     return {"message": "I am predictor"}
#
# @app.get("/api/composer")
# def compose():
#     return {"message": "I am composer"}
#
# @app.post("/api/predictor")
# def process(data: dict):
#     print("/api/predict: predict data: ")
#     print( data)
#     return {"result": 1}
#
# @app.post("/api/composer/midi")
# async def read_midi( midiFile: UploadFile = File(...) ):
#     content = await midiFile.read()
#
#     print("Received file:", midiFile.filename)
#     print("Size:", len(content))
#
#     # Convert bytes → file-like object (in memory)
#     midi_stream = io.BytesIO(content)
#
#     print("Parsing MIDI from memory...")
#
#     parser = Parser.MidiParser()
#     parser.read_midi(midi_stream)
#
#     # todo: call fine tune
#
#     return {"status": "ok"}
#
# @app.put("/api/composer/model/select")
# async def changeModel(data: ModelSelect): # jason data - model name
#     return {"model": data.model_name}
