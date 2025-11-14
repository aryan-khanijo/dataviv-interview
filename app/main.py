"""
FastAPI application for Real-Time Voice Translation + Lip-Synced Video Pipeline.
Provides endpoints for file upload, processing, status checking, and media retrieval.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.jobs import JobManager, JobStatus
from app.pipeline import run_pipeline
from app.storage import StorageManager

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global managers
storage_manager: Optional[StorageManager] = None
job_manager: Optional[JobManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    global storage_manager, job_manager
    storage_manager = StorageManager()
    job_manager = JobManager()
    logger.info("Application started")
    logger.info(f"Storage path: {settings.storage_path}")
    logger.info(f"Models path: {settings.models_path}")
    logger.info(f"Device: {settings.device}")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")


app = FastAPI(
    title="RTVT-LipSync API",
    description="Real-Time Voice Translation + Lip-Synced Video Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Pydantic Models
# ============================================================================

class UploadResponse(BaseModel):
    """Response model for file upload."""
    upload_id: str
    filename: str
    size: int
    status: str


class ChunkUploadResponse(BaseModel):
    """Response model for chunk upload."""
    upload_id: str
    chunk_index: int
    total_chunks: int
    status: str
    message: str


class ProcessRequest(BaseModel):
    """Request model for starting video processing."""
    upload_id: str = Field(..., description="Upload ID from the upload endpoint")
    target_lang: str = Field(
        ...,
        description="Target language code (e.g., 'es', 'fr', 'de')",
        pattern="^[a-z]{2}$",
    )
    source_lang: Optional[str] = Field(
        None,
        description="Source language code (auto-detected if not provided)",
    )


class ProcessResponse(BaseModel):
    """Response model for process initiation."""
    job_id: str
    status: str
    created_at: str


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    step: Optional[str] = None
    progress: int = 0
    message: Optional[str] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    media_id: Optional[str] = None
    steps: Optional[dict] = None


class JobListResponse(BaseModel):
    """Response model for job list."""
    jobs: list[JobStatusResponse]
    total: int
    limit: int
    offset: int


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "rtvt-lipsync",
        "version": "1.0.0",
    }


# ============================================================================
# Upload Endpoints
# ============================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
):
    """
    Upload a video or audio file.
    
    Supports files up to MAX_UPLOAD_SIZE (default 100MB).
    For larger files, use chunked upload endpoint.
    """
    try:
        # Check file size
        content = await file.read()
        if len(content) > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_upload_size} bytes",
            )
        
        # Save file
        upload_id = storage_manager.generate_upload_id()
        file_path = await storage_manager.save_upload(
            upload_id=upload_id,
            filename=file.filename,
            content=content,
        )
        
        logger.info(f"File uploaded: {upload_id} - {file.filename} ({len(content)} bytes)")
        
        return UploadResponse(
            upload_id=upload_id,
            filename=file.filename,
            size=len(content),
            status="complete",
        )
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/chunk", response_model=ChunkUploadResponse)
async def upload_chunk(
    file: UploadFile = File(...),
    upload_id: str = Header(...),
    chunk_index: int = Header(...),
    total_chunks: int = Header(...),
    filename: str = Header(...),
):
    """
    Upload a file chunk for large file uploads.
    
    Headers:
    - Upload-Id: Unique identifier for the upload session
    - Chunk-Index: Index of the current chunk (0-based)
    - Total-Chunks: Total number of chunks
    - Filename: Original filename
    """
    try:
        content = await file.read()
        
        # Save chunk
        is_complete = storage_manager.save_chunk(
            upload_id=upload_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            content=content,
        )
        
        if is_complete:
            # Assemble chunks
            final_path = await storage_manager.assemble_chunks(
                upload_id=upload_id,
                filename=filename,
            )
            logger.info(f"Chunked upload complete: {upload_id} - {filename}")
            
            return ChunkUploadResponse(
                upload_id=upload_id,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                status="complete",
                message="All chunks received and assembled",
            )
        else:
            return ChunkUploadResponse(
                upload_id=upload_id,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                status="pending",
                message=f"Chunk {chunk_index + 1}/{total_chunks} received",
            )
    
    except Exception as e:
        logger.error(f"Chunk upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/upload/{upload_id}/status")
async def get_upload_status(upload_id: str):
    """Get upload status and file information."""
    try:
        upload_info = storage_manager.get_upload_info(upload_id)
        if not upload_info:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return upload_info
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting upload status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Processing Endpoints
# ============================================================================

@app.post("/process", response_model=ProcessResponse)
async def start_processing(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start processing a video with translation and lip-sync.
    
    Creates a job and runs the pipeline asynchronously.
    Returns immediately with a job_id to track progress.
    """
    try:
        # Validate upload exists
        upload_info = storage_manager.get_upload_info(request.upload_id)
        if not upload_info:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        if upload_info["status"] != "complete":
            raise HTTPException(
                status_code=400,
                detail="Upload not complete. Wait for all chunks to be assembled.",
            )
        
        # Create job
        job = job_manager.create_job(
            upload_id=request.upload_id,
            target_lang=request.target_lang,
            source_lang=request.source_lang,
        )
        
        # Start pipeline in background
        background_tasks.add_task(
            run_pipeline_task,
            job_id=job["job_id"],
            upload_path=Path(upload_info["path"]),
            target_lang=request.target_lang,
            source_lang=request.source_lang,
        )
        
        logger.info(f"Job created: {job['job_id']} - Upload: {request.upload_id}")
        
        return ProcessResponse(
            job_id=job["job_id"],
            status=job["status"],
            created_at=job["created_at"],
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_pipeline_task(
    job_id: str,
    upload_path: Path,
    target_lang: str,
    source_lang: Optional[str] = None,
):
    """
    Background task wrapper for running the pipeline.
    """
    try:
        await run_pipeline(
            job_id=job_id,
            upload_path=upload_path,
            target_lang=target_lang,
            source_lang=source_lang,
            job_manager=job_manager,
            storage_manager=storage_manager,
        )
    except Exception as e:
        logger.error(f"Pipeline error for job {job_id}: {str(e)}")
        job_manager.update_job(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(e),
        )


@app.get("/process/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    try:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(**job)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all jobs with optional filtering."""
    try:
        jobs = job_manager.list_jobs(status=status, limit=limit, offset=offset)
        total = job_manager.count_jobs(status=status)
        
        return JobListResponse(
            jobs=[JobStatusResponse(**job) for job in jobs],
            total=total,
            limit=limit,
            offset=offset,
        )
    
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Media Endpoints
# ============================================================================

@app.get("/media/{media_id}")
async def get_media(media_id: str):
    """
    Download the processed video file.
    
    The media_id is typically the same as the job_id.
    """
    try:
        # Get job to find output file
        job = job_manager.get_job(media_id)
        if not job:
            raise HTTPException(status_code=404, detail="Media not found")
        
        if job["status"] != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Job not completed. Current status: {job['status']}",
            )
        
        if not job.get("media_id"):
            raise HTTPException(status_code=404, detail="Output file not found")
        
        # Get file path
        output_path = settings.outputs_dir / f"{job['media_id']}.mp4"
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found on disk")
        
        # Return file
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=f"translated_{job['upload_id']}.mp4",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving media: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"},
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
