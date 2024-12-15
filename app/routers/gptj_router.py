from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.services.gptj_service import GPTJService

router = APIRouter()
gptj_service = GPTJService()

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100

class GenerateResponse(BaseModel):
    generated_text: str

@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    generated_text = await gptj_service.generate_text(
        request.prompt,
        request.max_length
    )
    return GenerateResponse(generated_text=generated_text)