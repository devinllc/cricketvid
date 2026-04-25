"""
Cricket Video Assessment System — FastAPI Application Entry Point
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes import report as report_router
from app.routes import video as video_router
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Ensure required directories exist
for d in ["uploads", "processed", "static"]:
    Path(f"app/{d}").mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🏏 Cricket Video Assessment System starting up...")
    yield
    logger.info("🏏 Cricket Video Assessment System shutting down.")


app = FastAPI(
    title="Cricket Video Assessment System",
    description=(
        "AI-powered cricket practice video analysis. "
        "Upload batting videos and receive detailed technical assessment reports."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Serve processed artifacts (overlay videos, etc.)
processed_dir = Path(__file__).parent / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
app.mount("/processed", StaticFiles(directory=str(processed_dir)), name="processed")

# Templates
templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)

# Routers
app.include_router(video_router.router, tags=["Video Upload"])
app.include_router(report_router.router, tags=["Reports"])


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    """Serve the main upload UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "service": "Cricket Video Assessment System", "version": "1.0.0"}
