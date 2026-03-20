import os

from fastapi import FastAPI
from dotenv import load_dotenv

from api.routes import register_routes
from rag.config import load_config
from rag.store import VectorStoreManager


PROJECT_ROOT = os.path.dirname(__file__)

# Load environment variables from .env in the project root (if present)
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

cfg = load_config(PROJECT_ROOT)
store = VectorStoreManager(config=cfg)

app = FastAPI()
register_routes(app, cfg=cfg, store=store)
