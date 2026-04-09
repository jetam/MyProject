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
