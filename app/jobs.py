"""
Job management and state tracking.
Persists job metadata to JSON files with atomic updates.
"""

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStep(str, Enum):
    """Pipeline step enumeration."""
    ASR = "asr"
    TRANSLATE = "translate"
    TTS = "tts"
    LIPSYNC = "lipsync"


class JobManager:
    """Manages job creation, updates, and persistence."""
    
    def __init__(self):
        """Initialize job manager."""
        self.jobs_dir = settings.jobs_dir
        self.jobs_dir.mkdir(exist_ok=True)
    
    @staticmethod
    def generate_job_id() -> str:
        """Generate a unique job ID."""
        return str(uuid.uuid4())
    
    def create_job(
        self,
        upload_id: str,
        target_lang: str,
        source_lang: Optional[str] = None,
    ) -> Dict:
        """
        Create a new processing job.
        
        Args:
            upload_id: Upload identifier
            target_lang: Target language code
            source_lang: Source language code (optional)
        
        Returns:
            Job metadata dictionary
        """
        job_id = self.generate_job_id()
        now = datetime.utcnow().isoformat() + "Z"
        
        job = {
            "job_id": job_id,
            "upload_id": upload_id,
            "target_lang": target_lang,
            "source_lang": source_lang,
            "status": JobStatus.QUEUED,
            "step": None,
            "progress": 0,
            "message": "Job created, waiting to start",
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "error": None,
            "media_id": None,
            "steps": {
                JobStep.ASR: "pending",
                JobStep.TRANSLATE: "pending",
                JobStep.TTS: "pending",
                JobStep.LIPSYNC: "pending",
            },
        }
        
        self._save_job(job)
        logger.info(f"Created job: {job_id}")
        return job
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """
        Get job by ID.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job metadata dictionary or None if not found
        """
        job_file = self.jobs_dir / f"{job_id}.json"
        
        if not job_file.exists():
            return None
        
        try:
            with open(job_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading job {job_id}: {e}")
            return None
    
    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        step: Optional[JobStep] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        media_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Update job metadata.
        
        Args:
            job_id: Job identifier
            status: New status
            step: Current pipeline step
            progress: Progress percentage (0-100)
            message: Status message
            error: Error message (if failed)
            media_id: Media identifier (if completed)
        
        Returns:
            Updated job metadata or None if job not found
        """
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return None
        
        # Update fields
        if status is not None:
            job["status"] = status
        
        if step is not None:
            job["step"] = step
            # Update step status
            if step in job["steps"]:
                job["steps"][step] = "processing"
        
        if progress is not None:
            job["progress"] = min(100, max(0, progress))
        
        if message is not None:
            job["message"] = message
        
        if error is not None:
            job["error"] = error
        
        if media_id is not None:
            job["media_id"] = media_id
        
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        if status == JobStatus.COMPLETED:
            job["completed_at"] = job["updated_at"]
            job["progress"] = 100
            # Mark all steps as completed
            for step_key in job["steps"]:
                job["steps"][step_key] = "completed"
        
        if status == JobStatus.FAILED:
            job["completed_at"] = job["updated_at"]
        
        self._save_job(job)
        logger.debug(f"Updated job {job_id}: {status} - {message}")
        return job
    
    def update_step(
        self,
        job_id: str,
        step: JobStep,
        status: str,
        progress: Optional[int] = None,
        message: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Update a specific pipeline step.
        
        Args:
            job_id: Job identifier
            step: Pipeline step
            status: Step status (pending, processing, completed, failed)
            progress: Overall progress percentage
            message: Status message
        
        Returns:
            Updated job metadata or None if job not found
        """
        job = self.get_job(job_id)
        if not job:
            return None
        
        # Update step status
        if step in job["steps"]:
            job["steps"][step] = status
        
        # Update overall status and progress
        if progress is not None:
            job["progress"] = progress
        
        if message is not None:
            job["message"] = message
        
        job["step"] = step
        job["status"] = JobStatus.PROCESSING
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        self._save_job(job)
        return job
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict]:
        """
        List jobs with optional filtering.
        
        Args:
            status: Filter by status
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
        
        Returns:
            List of job metadata dictionaries
        """
        job_files = sorted(
            self.jobs_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        jobs = []
        for job_file in job_files:
            try:
                with open(job_file, "r") as f:
                    job = json.load(f)
                
                # Filter by status
                if status and job.get("status") != status:
                    continue
                
                jobs.append(job)
            
            except Exception as e:
                logger.error(f"Error loading job file {job_file}: {e}")
                continue
        
        # Apply pagination
        return jobs[offset:offset + limit]
    
    def count_jobs(self, status: Optional[str] = None) -> int:
        """
        Count jobs with optional filtering.
        
        Args:
            status: Filter by status
        
        Returns:
            Number of jobs
        """
        if status:
            jobs = self.list_jobs(status=status, limit=999999)
            return len(jobs)
        else:
            return len(list(self.jobs_dir.glob("*.json")))
    
    def _save_job(self, job: Dict):
        """
        Save job metadata to disk atomically.
        
        Args:
            job: Job metadata dictionary
        """
        job_id = job["job_id"]
        job_file = self.jobs_dir / f"{job_id}.json"
        temp_file = self.jobs_dir / f".{job_id}.json.tmp"
        
        try:
            # Write to temp file first
            with open(temp_file, "w") as f:
                json.dump(job, f, indent=2)
            
            # Atomic rename
            temp_file.replace(job_file)
        
        except Exception as e:
            logger.error(f"Error saving job {job_id}: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise


# Redis-based job manager (optional, for reference)
# Uncomment and implement when using Redis

"""
import redis
from redis import Redis

class RedisJobManager:
    '''Redis-based job manager for distributed systems.'''
    
    def __init__(self):
        self.redis: Redis = redis.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
        self.job_prefix = "job:"
        self.job_list_key = "jobs:all"
    
    def create_job(self, upload_id: str, target_lang: str, source_lang: Optional[str] = None) -> Dict:
        '''Create job in Redis.'''
        job_id = self.generate_job_id()
        now = datetime.utcnow().isoformat() + "Z"
        
        job = {
            "job_id": job_id,
            "upload_id": upload_id,
            "target_lang": target_lang,
            "source_lang": source_lang,
            "status": JobStatus.QUEUED,
            "created_at": now,
            "updated_at": now,
        }
        
        # Store job
        key = f"{self.job_prefix}{job_id}"
        self.redis.hset(key, mapping=job)
        
        # Add to list
        self.redis.zadd(self.job_list_key, {job_id: datetime.utcnow().timestamp()})
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        '''Get job from Redis.'''
        key = f"{self.job_prefix}{job_id}"
        data = self.redis.hgetall(key)
        
        if not data:
            return None
        
        return data
    
    def update_job(self, job_id: str, **kwargs) -> Optional[Dict]:
        '''Update job in Redis.'''
        key = f"{self.job_prefix}{job_id}"
        
        if not self.redis.exists(key):
            return None
        
        kwargs["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.redis.hset(key, mapping=kwargs)
        
        return self.get_job(job_id)
"""
