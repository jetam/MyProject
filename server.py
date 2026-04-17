# server.py

from fastapi import FastAPI
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

# rest:
# | Method | Meaning |
# | ------ | ------- |
# | GET    | read    |
# | POST   | create  |
# | PUT    | replace |
# | PATCH  | modify  |
# | DELETE | remove  |
