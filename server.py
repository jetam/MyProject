
from backend.pg_connection import Connection
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

@app.get("/api/predictor")
def predict():
    return {"message": "I am predictor"}

@app.get("/api/composer")
def predict():
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
async def upload_midi( midiFile: UploadFile = File(...) ):
    print("/api/compose/midi: upload midi data: ")
    content = await midiFile.read()

    print("Received file:", midiFile.filename)
    print("Size:", len(content))

    dbConnection = Connection( "localhost", "myproject", "matej", "postgres12345" )

    dbConnection.connect()

    dbConnection.close()

    return {"status": "ok"}

# rest:
# | Method | Meaning |
# | ------ | ------- |
# | GET    | read    |
# | POST   | create  |
# | PUT    | replace |
# | PATCH  | modify  |
# | DELETE | remove  |
