"""
Utility functions for audio/video processing using FFmpeg.
Provides wrappers for common operations like extraction, concatenation, and combination.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import ffmpeg
import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)


# ============================================================================
# Audio Extraction
# ============================================================================

def extract_audio(
    video_path: Path,
    audio_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Extract audio from video file.
    
    Args:
        video_path: Path to input video
        audio_path: Path to output audio file
        sample_rate: Audio sample rate (Hz)
        channels: Number of audio channels (1=mono, 2=stereo)
    
    Returns:
        Path to extracted audio file
    """
    try:
        logger.info(f"Extracting audio from {video_path}")
        
        # Use ffmpeg-python for clean API
        stream = ffmpeg.input(str(video_path))
        stream = ffmpeg.output(
            stream,
            str(audio_path),
            acodec='pcm_s16le',
            ac=channels,
            ar=sample_rate,
        )
        
        # Run with overwrite
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        logger.info(f"Audio extracted: {audio_path}")
        return audio_path
    
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        raise RuntimeError(f"Failed to extract audio: {str(e)}")


# ============================================================================
# Audio Concatenation
# ============================================================================

def concatenate_audio_segments(
    audio_files: List[Path],
    start_times: List[float],
    end_times: List[float],
    output_path: Path,
    sample_rate: int = 16000,
) -> Path:
    """
    Concatenate audio segments with proper timing alignment.
    
    Inserts silence between segments to maintain original timing.
    
    Args:
        audio_files: List of audio file paths
        start_times: Start time for each segment (seconds)
        end_times: End time for each segment (seconds)
        output_path: Output audio file path
        sample_rate: Audio sample rate
    
    Returns:
        Path to concatenated audio file
    """
    try:
        logger.info(f"Concatenating {len(audio_files)} audio segments")
        
        # Load all audio segments
        segments = []
        for audio_file in audio_files:
            rate, data = wavfile.read(audio_file)
            
            # Resample if needed
            if rate != sample_rate:
                # Simple resampling (use librosa for better quality)
                data = _resample_audio(data, rate, sample_rate)
            
            segments.append(data)
        
        # Build final audio with proper timing
        total_duration = end_times[-1]
        total_samples = int(total_duration * sample_rate)
        output_audio = np.zeros(total_samples, dtype=np.int16)
        
        for i, (segment, start, end) in enumerate(zip(segments, start_times, end_times)):
            start_sample = int(start * sample_rate)
            segment_length = len(segment)
            end_sample = min(start_sample + segment_length, total_samples)
            
            # Place segment at correct position
            output_audio[start_sample:end_sample] = segment[:end_sample - start_sample]
        
        # Save output
        wavfile.write(output_path, sample_rate, output_audio)
        
        logger.info(f"Audio concatenated: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error concatenating audio: {e}")
        raise


def _resample_audio(data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """
    Simple audio resampling (for better quality, use librosa).
    
    Args:
        data: Audio data
        orig_rate: Original sample rate
        target_rate: Target sample rate
    
    Returns:
        Resampled audio data
    """
    duration = len(data) / orig_rate
    target_length = int(duration * target_rate)
    
    # Linear interpolation (basic resampling)
    indices = np.linspace(0, len(data) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(data)), data)
    
    return resampled.astype(data.dtype)


# ============================================================================
# Audio/Video Combination
# ============================================================================

def combine_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    video_codec: str = "copy",
    audio_codec: str = "aac",
) -> Path:
    """
    Combine video with new audio track.
    
    Args:
        video_path: Path to input video
        audio_path: Path to input audio
        output_path: Path to output video
        video_codec: Video codec (default: copy - faster, no re-encoding)
        audio_codec: Audio codec (default: aac)
    
    Returns:
        Path to output video
    """
    try:
        logger.info(f"Combining video {video_path} with audio {audio_path}")
        
        # Input streams
        video = ffmpeg.input(str(video_path))
        audio = ffmpeg.input(str(audio_path))
        
        # Combine streams - map video from first input and audio from second
        stream = ffmpeg.output(
            video.video,
            audio.audio,
            str(output_path),
            vcodec=video_codec,
            acodec=audio_codec,
            audio_bitrate='192k',
            strict='experimental',
        )
        
        # Run with error capture
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        logger.info(f"Combined video created: {output_path}")
        return output_path
    
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error: {error_msg}")
        raise RuntimeError(f"Failed to combine audio/video: {error_msg}")


# ============================================================================
# Stub Utilities
# ============================================================================

def create_silent_audio(
    output_path: Path,
    duration: float,
    sample_rate: int = 16000,
) -> Path:
    """
    Create a silent audio file of specified duration.
    
    Useful for testing and stub implementations.
    
    Args:
        output_path: Path to output audio file
        duration: Duration in seconds
        sample_rate: Audio sample rate
    
    Returns:
        Path to created audio file
    """
    try:
        logger.info(f"Creating silent audio: {duration}s")
        
        # Generate silent audio
        samples = int(duration * sample_rate)
        silent_audio = np.zeros(samples, dtype=np.int16)
        
        # Save
        wavfile.write(output_path, sample_rate, silent_audio)
        
        logger.info(f"Silent audio created: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating silent audio: {e}")
        raise


# ============================================================================
# Video Processing Utilities
# ============================================================================

def get_video_info(video_path: Path) -> dict:
    """
    Get video metadata using ffprobe.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video metadata
    """
    try:
        probe = ffmpeg.probe(str(video_path))
        
        # Extract video stream info
        video_stream = next(
            (s for s in probe['streams'] if s['codec_type'] == 'video'),
            None
        )
        
        # Extract audio stream info
        audio_stream = next(
            (s for s in probe['streams'] if s['codec_type'] == 'audio'),
            None
        )
        
        info = {
            'format': probe['format']['format_name'],
            'duration': float(probe['format'].get('duration', 0)),
            'size': int(probe['format'].get('size', 0)),
        }
        
        if video_stream:
            info.update({
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                'video_codec': video_stream['codec_name'],
            })
        
        if audio_stream:
            info.update({
                'audio_codec': audio_stream['codec_name'],
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
            })
        
        return info
    
    except ffmpeg.Error as e:
        logger.error(f"Error probing video: {e}")
        raise


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: Optional[int] = None,
) -> List[Path]:
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Frames per second to extract (None = all frames)
    
    Returns:
        List of frame file paths
    """
    try:
        logger.info(f"Extracting frames from {video_path}")
        output_dir.mkdir(exist_ok=True)
        
        # Build output pattern
        output_pattern = str(output_dir / "frame_%05d.jpg")
        
        # Build command
        stream = ffmpeg.input(str(video_path))
        
        if fps:
            stream = ffmpeg.filter(stream, 'fps', fps=fps)
        
        stream = ffmpeg.output(stream, output_pattern)
        
        # Run
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        # Get list of created frames
        frames = sorted(output_dir.glob("frame_*.jpg"))
        
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    except ffmpeg.Error as e:
        logger.error(f"Error extracting frames: {e}")
        raise


def create_video_from_frames(
    frames_dir: Path,
    output_path: Path,
    fps: int = 25,
    codec: str = "libx264",
) -> Path:
    """
    Create video from frame images.
    
    Args:
        frames_dir: Directory containing frame images
        output_path: Path to output video
        fps: Frame rate
        codec: Video codec
    
    Returns:
        Path to created video
    """
    try:
        logger.info(f"Creating video from frames in {frames_dir}")
        
        # Input pattern
        input_pattern = str(frames_dir / "frame_%05d.jpg")
        
        # Build command
        stream = ffmpeg.input(input_pattern, framerate=fps)
        stream = ffmpeg.output(stream, str(output_path), vcodec=codec)
        
        # Run
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        logger.info(f"Video created: {output_path}")
        return output_path
    
    except ffmpeg.Error as e:
        logger.error(f"Error creating video: {e}")
        raise


# ============================================================================
# Audio Cutting
# ============================================================================

def cut_audio(
    input_path: Path,
    output_path: Path,
    start_time: float,
    duration: Optional[float] = None,
) -> Path:
    """
    Cut audio segment from file.
    
    Args:
        input_path: Path to input audio
        output_path: Path to output audio
        start_time: Start time in seconds
        duration: Duration in seconds (None = to end)
    
    Returns:
        Path to cut audio file
    """
    try:
        stream = ffmpeg.input(str(input_path), ss=start_time)
        
        if duration:
            stream = ffmpeg.output(stream, str(output_path), t=duration)
        else:
            stream = ffmpeg.output(stream, str(output_path))
        
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        return output_path
    
    except ffmpeg.Error as e:
        logger.error(f"Error cutting audio: {e}")
        raise


# ============================================================================
# Path Utilities
# ============================================================================

def ensure_path_safe(path: Path, base_dir: Path) -> Path:
    """
    Ensure path is within base directory (prevent path traversal).
    
    Args:
        path: Path to check
        base_dir: Base directory
    
    Returns:
        Resolved safe path
    
    Raises:
        ValueError: If path is outside base directory
    """
    resolved_path = path.resolve()
    resolved_base = base_dir.resolve()
    
    if not str(resolved_path).startswith(str(resolved_base)):
        raise ValueError(f"Path {path} is outside base directory {base_dir}")
    
    return resolved_path
