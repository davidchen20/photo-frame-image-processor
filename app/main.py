from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from fastapi import FastAPI, BackgroundTasks, Header, HTTPException, status
from pydantic import BaseModel

from app.config import get_settings
from app.services.processor import ensure_model
from app.services.pipeline import run_image_pipeline

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # make sure model is downloaded
    ensure_model()
    yield
    
app = FastAPI(
    title="Photo Frame Cloud Microservice",
    version="1.0.0",
    lifespan=lifespan
)

class StorageRecord(BaseModel):
    name: str
    bucket_id: str
    metadata: Optional[Dict[str, Any]] = None
    
class SupabaseWebhookPaylod(BaseModel):
    type: str
    table: str
    schema: str
    record: StorageRecord
    
@app.get("/heartbeat")
async def heartbeat():
    return {"status": "ok"}

@app.post("/api/v1/storage-webhook", status_code=status.HTTP_202_ACCEPTED)
async def handle_storage_webhook(paylod: SupabaseWebhookPaylod, background_tasks: BackgroundTasks, x_webhook_secret: Optional[str] = Header(None)):
    if x_webhook_secret != settings.WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing x_webhook_secret"
        )
        
    if paylod.type == "INSERT" and paylod.record.bucket_id == settings.BUCKET_NAME:
        file_path = paylod.record.name
        
        background_tasks.add_task(run_image_pipeline, file_path=file_path)
        return {"status": "queued", "file": file_path}
    
    return {"status": "skipped", "reason": "not an insert"}
        
    