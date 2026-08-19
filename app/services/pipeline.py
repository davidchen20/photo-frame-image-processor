import asyncio
import os
from app.config import get_settings
from app.core.supabase import get_supabase
from app.services.processor import smart_crop

async def run_image_pipeline(file_path: str) -> None:
    settings = get_settings()
    supabase = get_supabase()
    
    # don't need to process iamges in the processed folder again
    if file_path.startswith("processed/"):
        return
    
    try:
        raw_bytes = supabase.storage.from_(settings.BUCKET_NAME).download(file_path)

        # process the img
        target_size = (settings.TARGET_WIDTH, settings.TARGET_HEIGHT)
        processed_bytes = await asyncio.to_thread(smart_crop, raw_bytes=raw_bytes, target_size=target_size)

        # save it to the processed folder as a .bmp
        filename = os.path.basename(file_path)
        base_name, _ = os.path.splitext(filename)
        destination_path = f"processed/{base_name}.bmp"
        supabase.storage.from_(settings.BUCKET_NAME).upload(
            path=destination_path,
            file=processed_bytes,
            file_options={"content-type": "image/bmp", "upsert": "true"},
        )
        
        # delete it from uploads folder since im on free tier and only have 1 gb
        supabase.storage.from_(settings.BUCKET_NAME).remove([file_path])
    except Exception as e:
        print(f"Got an exception while processing: {e}")
        