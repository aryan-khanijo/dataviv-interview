"""
Storage management for file uploads and outputs.
Handles chunked uploads, file assembly, and local disk storage.
"""

import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages file storage, uploads, and chunked upload assembly."""
    
    def __init__(self):
        """Initialize storage manager."""
        self.uploads_dir = settings.uploads_dir
        self.processing_dir = settings.processing_dir
        self.outputs_dir = settings.outputs_dir
        self.chunks_dir = settings.storage_path / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        
        # In-memory tracking for chunked uploads
        self.chunk_status: Dict[str, Dict] = {}
    
    @staticmethod
    def generate_upload_id() -> str:
        """Generate a unique upload ID."""
        return str(uuid.uuid4())
    
    async def save_upload(
        self,
        upload_id: str,
        filename: str,
        content: bytes,
    ) -> Path:
        """
        Save uploaded file content to disk.
        
        Args:
            upload_id: Unique upload identifier
            filename: Original filename
            content: File content bytes
        
        Returns:
            Path to saved file
        """
        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)
        file_path = self.uploads_dir / f"{upload_id}_{safe_filename}"
        
        # Write file asynchronously
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._write_file,
            file_path,
            content,
        )
        
        logger.info(f"Saved upload: {file_path}")
        return file_path
    
    def save_chunk(
        self,
        upload_id: str,
        chunk_index: int,
        total_chunks: int,
        content: bytes,
    ) -> bool:
        """
        Save a file chunk and track progress.
        
        Args:
            upload_id: Unique upload identifier
            chunk_index: Index of this chunk (0-based)
            total_chunks: Total number of chunks expected
            content: Chunk content bytes
        
        Returns:
            True if all chunks received, False otherwise
        """
        # Create chunk directory for this upload
        upload_chunks_dir = self.chunks_dir / upload_id
        upload_chunks_dir.mkdir(exist_ok=True)
        
        # Save chunk
        chunk_path = upload_chunks_dir / f"chunk_{chunk_index:05d}"
        self._write_file(chunk_path, content)
        
        # Track progress
        if upload_id not in self.chunk_status:
            self.chunk_status[upload_id] = {
                "total": total_chunks,
                "received": set(),
            }
        
        self.chunk_status[upload_id]["received"].add(chunk_index)
        
        # Check if complete
        received = len(self.chunk_status[upload_id]["received"])
        is_complete = received == total_chunks
        
        logger.debug(f"Chunk {chunk_index}/{total_chunks} saved for {upload_id}")
        
        return is_complete
    
    async def assemble_chunks(
        self,
        upload_id: str,
        filename: str,
    ) -> Path:
        """
        Assemble all chunks into a single file.
        
        Args:
            upload_id: Unique upload identifier
            filename: Original filename
        
        Returns:
            Path to assembled file
        """
        upload_chunks_dir = self.chunks_dir / upload_id
        
        if not upload_chunks_dir.exists():
            raise ValueError(f"No chunks found for upload {upload_id}")
        
        # Get chunk status
        status = self.chunk_status.get(upload_id)
        if not status:
            raise ValueError(f"No chunk tracking for upload {upload_id}")
        
        total_chunks = status["total"]
        
        # Assemble chunks
        safe_filename = self._sanitize_filename(filename)
        output_path = self.uploads_dir / f"{upload_id}_{safe_filename}"
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._assemble_chunks_sync,
            upload_chunks_dir,
            output_path,
            total_chunks,
        )
        
        # Cleanup chunks
        shutil.rmtree(upload_chunks_dir, ignore_errors=True)
        del self.chunk_status[upload_id]
        
        logger.info(f"Assembled {total_chunks} chunks into {output_path}")
        return output_path
    
    def _assemble_chunks_sync(
        self,
        chunks_dir: Path,
        output_path: Path,
        total_chunks: int,
    ):
        """Synchronous chunk assembly."""
        with open(output_path, "wb") as outfile:
            for i in range(total_chunks):
                chunk_path = chunks_dir / f"chunk_{i:05d}"
                if not chunk_path.exists():
                    raise ValueError(f"Missing chunk {i}")
                
                with open(chunk_path, "rb") as chunk_file:
                    shutil.copyfileobj(chunk_file, outfile)
    
    def get_upload_info(self, upload_id: str) -> Optional[Dict]:
        """
        Get information about an upload.
        
        Args:
            upload_id: Upload identifier
        
        Returns:
            Dictionary with upload info or None if not found
        """
        # Check for complete upload
        uploads = list(self.uploads_dir.glob(f"{upload_id}_*"))
        if uploads:
            file_path = uploads[0]
            return {
                "upload_id": upload_id,
                "filename": file_path.name.replace(f"{upload_id}_", "", 1),
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "status": "complete",
            }
        
        # Check for pending chunked upload
        if upload_id in self.chunk_status:
            status = self.chunk_status[upload_id]
            return {
                "upload_id": upload_id,
                "status": "pending",
                "chunks_received": len(status["received"]),
                "total_chunks": status["total"],
            }
        
        return None
    
    def get_processing_path(self, job_id: str) -> Path:
        """
        Get processing directory for a job.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Path to job processing directory
        """
        job_dir = self.processing_dir / job_id
        job_dir.mkdir(exist_ok=True)
        return job_dir
    
    def save_output(
        self,
        job_id: str,
        output_path: Path,
    ) -> Path:
        """
        Save final output file.
        
        Args:
            job_id: Job identifier
            output_path: Path to output file
        
        Returns:
            Path to saved output file
        """
        final_path = self.outputs_dir / f"{job_id}.mp4"
        shutil.copy2(output_path, final_path)
        logger.info(f"Saved output: {final_path}")
        return final_path
    
    def cleanup_processing(self, job_id: str):
        """
        Clean up processing files for a job.
        
        Args:
            job_id: Job identifier
        """
        job_dir = self.processing_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.info(f"Cleaned up processing files for job {job_id}")
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and issues.
        
        Args:
            filename: Original filename
        
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = Path(filename).name
        
        # Replace problematic characters
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        sanitized = "".join(c if c in safe_chars else "_" for c in filename)
        
        # Ensure not empty
        if not sanitized or sanitized.startswith("."):
            sanitized = f"file_{uuid.uuid4().hex[:8]}"
        
        return sanitized
    
    @staticmethod
    def _write_file(path: Path, content: bytes):
        """Write content to file (blocking I/O)."""
        with open(path, "wb") as f:
            f.write(content)


# MinIO implementation (optional, for reference)
# Uncomment and implement when using MinIO

"""
from minio import Minio
from minio.error import S3Error

class MinIOStorageManager:
    '''MinIO-based storage manager.'''
    
    def __init__(self):
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket = settings.minio_bucket
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        '''Ensure bucket exists.'''
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error as e:
            logger.error(f"MinIO error: {e}")
            raise
    
    async def save_upload(self, upload_id: str, filename: str, content: bytes) -> str:
        '''Upload file to MinIO.'''
        object_name = f"uploads/{upload_id}_{filename}"
        
        # MinIO client is synchronous, run in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.client.put_object,
            self.bucket,
            object_name,
            io.BytesIO(content),
            len(content),
        )
        
        return object_name
    
    async def get_object(self, object_name: str) -> bytes:
        '''Download object from MinIO.'''
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.client.get_object,
            self.bucket,
            object_name,
        )
        
        return response.read()
"""
