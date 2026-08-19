"""Give access to a supabase client to allow reads and writes to supabase container"""

from supabase import create_client, Client
from app.config import get_settings

supabase_client : Client | None = None

def get_supabase() -> Client:
    global supabase_client
    if supabase_client is None:
        settings = get_settings()
        supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
        
    return supabase_client