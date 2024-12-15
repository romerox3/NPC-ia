from fastapi import FastAPI
from app.routers import gptj_router

app = FastAPI(title="GPT-J API")

app.include_router(gptj_router.router, prefix="/api/v1", tags=["gptj"])

@app.get("/health")
async def health_check():
    return {"status": "ok"}