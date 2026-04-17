# server.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# api/compose
# api/predict

@app.get("/api/predict")
def predict():
    return {"message": "I am predictor"}

@app.get("/api/compose")
def predict():
    return {"message": "I am composer"}

@app.post("/api/predict")
def process(data: dict):
    print("/api/predict: predict data: ")
    print( data)  # see what frontend sent
    return {"result": 1}

@app.post("/api/compose")
def process(data: dict):
    print("/api/compose: compose data: ")
    print( data)  # see what frontend sent
    return {"result": 2}

@app.post("/api/compose/midi")
async def upload_midi( midiFile: UploadFile = File(...) ):
    print("/api/compose/midi: upload midi data: ")
    content = await midiFile.read()

    print("Received file:", midiFile.filename)
    print("Size:", len(content))

    return {"status": "ok"}

# rest:
# | Method | Meaning |
# | ------ | ------- |
# | GET    | read    |
# | POST   | create  |
# | PUT    | replace |
# | PATCH  | modify  |
# | DELETE | remove  |
