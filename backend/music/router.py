from fastapi import APIRouter, File, Response, UploadFile
from pydantic import BaseModel

from .services.composer import composer # import instance of Composer
from .services.composer import ModelNames

router = APIRouter()

class ModelSelection(BaseModel):
    model_name: str

class GenerateRequest(BaseModel):
    midi_id: int

@router.post("/midi/upload")
async def upload_midi(midiFile: UploadFile = File(...)):
    print("Received file:", midiFile.filename)
    content = await midiFile.read()
    composer.handleUpload(content) # fine tune and save model
    return {"status": "ok"}

@router.post("/generate")
async def generate_music():
    if composer.currentModelName == "Not Selected":
        return {"status": "Model Not Selected"}
    midi_bytes = composer.generateMusic()
    return Response(
        content=midi_bytes,
        media_type="audio/midi",
        headers={"Content-Disposition": "attachment; filename=generated.mid"},
    )

@router.put("/model/select")
async def select_model(selection: ModelSelection):
    composer.selectModel(ModelNames(selection.model_name))
    return {"message": "Model selected successfully"}

@router.get("/model/selection") # sends all possible models to frontend
async def models():
    return {"models": ModelNames.allModels()}
