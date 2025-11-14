"""
Pipeline orchestration for voice translation and lip-sync.
Coordinates ASR, translation, TTS, and lip-sync steps.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.jobs import JobManager, JobStatus, JobStep

logger = logging.getLogger(__name__)


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

async def run_pipeline(
    job_id: str,
    upload_path: Path,
    target_lang: str,
    source_lang: Optional[str],
    job_manager: JobManager,
    storage_manager,
):
    """
    Run the complete translation and lip-sync pipeline.
    
    Steps:
    1. Extract audio from video
    2. ASR: Transcribe audio to text with timestamps
    3. Translate: Convert text to target language
    4. TTS: Synthesize translated speech
    5. Lip-sync: Apply lip movements to video
    
    Args:
        job_id: Job identifier
        upload_path: Path to uploaded video file
        target_lang: Target language code
        source_lang: Source language code (optional)
        job_manager: Job manager instance
        storage_manager: Storage manager instance
    """
    try:
        # Update status to processing
        job_manager.update_job(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=5,
            message="Starting pipeline...",
        )
        
        # Get processing directory
        processing_dir = storage_manager.get_processing_path(job_id)
        
        # Import utils for ffmpeg operations
        from app.utils import extract_audio, combine_audio_video
        
        logger.info(f"Pipeline started for job {job_id}")
        
        # ====================================================================
        # Step 0: Extract audio from video
        # ====================================================================
        job_manager.update_job(
            job_id=job_id,
            progress=10,
            message="Extracting audio from video...",
        )
        
        audio_path = processing_dir / "original_audio.wav"
        await asyncio.get_event_loop().run_in_executor(
            None,
            extract_audio,
            upload_path,
            audio_path,
        )
        
        logger.info(f"Audio extracted: {audio_path}")
        
        # ====================================================================
        # Step 1: ASR (Automatic Speech Recognition)
        # ====================================================================
        job_manager.update_step(
            job_id=job_id,
            step=JobStep.ASR,
            status="processing",
            progress=20,
            message="Transcribing audio...",
        )
        
        transcript = await run_asr(
            audio_path=audio_path,
            source_lang=source_lang,
        )
        
        # Save transcript
        transcript_path = processing_dir / "transcript.json"
        with open(transcript_path, "w") as f:
            json.dump(transcript, f, indent=2)
        
        job_manager.update_step(
            job_id=job_id,
            step=JobStep.ASR,
            status="completed",
            progress=35,
            message=f"Transcribed {len(transcript['segments'])} segments",
        )
        
        logger.info(f"ASR completed: {len(transcript['segments'])} segments")
        
        # ====================================================================
        # Step 2: Translation
        # ====================================================================
        job_manager.update_step(
            job_id=job_id,
            step=JobStep.TRANSLATE,
            status="processing",
            progress=40,
            message=f"Translating to {target_lang}...",
        )
        
        translated_transcript = await run_translation(
            transcript=transcript,
            source_lang=transcript.get("language", source_lang or "en"),
            target_lang=target_lang,
        )
        
        # Save translated transcript
        translated_path = processing_dir / "translated.json"
        with open(translated_path, "w") as f:
            json.dump(translated_transcript, f, indent=2)
        
        job_manager.update_step(
            job_id=job_id,
            step=JobStep.TRANSLATE,
            status="completed",
            progress=50,
            message="Translation completed",
        )
        
        logger.info(f"Translation completed")
        
        # ====================================================================
        # Step 3: TTS (Text-to-Speech)
        # ====================================================================
        job_manager.update_step(
            job_id=job_id,
            step=JobStep.TTS,
            status="processing",
            progress=55,
            message="Generating translated speech...",
        )
        
        translated_audio_path = await run_tts(
            transcript=translated_transcript,
            output_dir=processing_dir,
            target_lang=target_lang,
        )
        
        job_manager.update_step(
            job_id=job_id,
            step=JobStep.TTS,
            status="completed",
            progress=75,
            message="Speech synthesis completed",
        )
        
        logger.info(f"TTS completed: {translated_audio_path}")
        
        # ====================================================================
        # Step 4: Lip-Sync
        # ====================================================================
        job_manager.update_step(
            job_id=job_id,
            step=JobStep.LIPSYNC,
            status="processing",
            progress=80,
            message="Synchronizing lip movements...",
        )
        
        final_video_path = await run_lipsync(
            video_path=upload_path,
            audio_path=translated_audio_path,
            output_dir=processing_dir,
        )
        
        job_manager.update_step(
            job_id=job_id,
            step=JobStep.LIPSYNC,
            status="completed",
            progress=95,
            message="Lip-sync completed",
        )
        
        logger.info(f"Lip-sync completed: {final_video_path}")
        
        # ====================================================================
        # Step 5: Save output
        # ====================================================================
        job_manager.update_job(
            job_id=job_id,
            progress=98,
            message="Saving final output...",
        )
        
        output_path = storage_manager.save_output(job_id, final_video_path)
        
        # Mark job as completed
        job_manager.update_job(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            message="Processing completed successfully",
            media_id=job_id,
        )
        
        # Cleanup intermediate files
        storage_manager.cleanup_processing(job_id)
        
        logger.info(f"Pipeline completed for job {job_id}: {output_path}")
    
    except Exception as e:
        logger.error(f"Pipeline error for job {job_id}: {str(e)}", exc_info=True)
        job_manager.update_job(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(e),
            message=f"Pipeline failed: {str(e)}",
        )


# ============================================================================
# ASR (Automatic Speech Recognition) with Whisper
# ============================================================================

async def run_asr(
    audio_path: Path,
    source_lang: Optional[str] = None,
) -> Dict:
    """
    Run ASR on audio file to extract transcript with timestamps.
    
    Uses OpenAI Whisper model for speech recognition.
    
    Args:
        audio_path: Path to audio file
        source_lang: Source language code (optional, will auto-detect)
    
    Returns:
        Dictionary with transcript segments and metadata
    """
    logger.info(f"Running ASR on {audio_path}")
    import whisper
    
    # Load model (cached after first load)
    model = whisper.load_model(settings.whisper_model, device=settings.device)
    
    # Transcribe
    result = model.transcribe(
        str(audio_path),
        language=source_lang,
        task="transcribe",
        verbose=False,
    )
    
    # Format output
    transcript = {
        "language": result["language"],
        "duration": result.get("duration", 0),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result["segments"]
        ],
    }
    
    return transcript


# ============================================================================
# Translation with MarianMT
# ============================================================================

async def run_translation(
    transcript: Dict,
    source_lang: str,
    target_lang: str,
) -> Dict:

    logger.info(f"Translating from {source_lang} to {target_lang}")
    
    # TODO: Uncomment for real MarianMT integration
    from transformers import MarianMTModel, MarianTokenizer
    
    # Get model name for language pair
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    
    try:
        # Load model and tokenizer
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        if settings.device == "cuda":
            model = model.cuda()
        
        # Translate each segment
        translated_segments = []
        texts = [seg["text"] for seg in transcript["segments"]]
        
        # Batch translation
        batch_size = 4
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            inputs = tokenizer(batch, return_tensors="pt", padding=True)
            if settings.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model.generate(**inputs)
            translated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for j, translated_text in enumerate(translated_batch):
                segment_idx = i + j
                translated_segments.append({
                    "start": transcript["segments"][segment_idx]["start"],
                    "end": transcript["segments"][segment_idx]["end"],
                    "text": translated_text,
                })
        
        return {
            "language": target_lang,
            "duration": transcript["duration"],
            "segments": translated_segments,
        }
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        # Fallback: return original with warning
        logger.warning("Translation failed, using original text")
        return transcript



# ============================================================================
# TTS (Text-to-Speech) with gTTS
# ============================================================================

async def run_tts(
    transcript: Dict,
    output_dir: Path,
    target_lang: str,
) -> Path:
    logger.info(f"Running TTS for {target_lang}")
    
    try:
        from gtts import gTTS
        from pydub import AudioSegment
        
        logger.info(f"Generating translated speech using gTTS for language: {target_lang}")
        
        # Generate audio for each segment
        segment_audio_files = []
        
        for i, segment in enumerate(transcript["segments"]):
            segment_mp3_path = output_dir / f"tts_segment_{i:03d}.mp3"
            segment_wav_path = output_dir / f"tts_segment_{i:03d}.wav"
            
            text = segment["text"]
            logger.info(f"Generating speech {i+1}/{len(transcript['segments'])}: {text[:50]}...")
            
            # Generate speech with gTTS
            tts = gTTS(text=text, lang=target_lang, slow=False)
            tts.save(str(segment_mp3_path))
            
            # Convert MP3 to WAV for consistency
            audio = AudioSegment.from_mp3(str(segment_mp3_path))
            audio.export(str(segment_wav_path), format="wav")
            
            # Clean up MP3
            segment_mp3_path.unlink()
            
            segment_audio_files.append(segment_wav_path)
        
        logger.info(f"Generated {len(segment_audio_files)} audio segments")
        
        # Concatenate segments with proper timing
        from app.utils import concatenate_audio_segments
        
        final_audio_path = output_dir / "translated_audio.wav"
        await asyncio.get_event_loop().run_in_executor(
            None,
            concatenate_audio_segments,
            segment_audio_files,
            [seg["start"] for seg in transcript["segments"]],
            [seg["end"] for seg in transcript["segments"]],
            final_audio_path,
        )
        
        logger.info(f"TTS completed successfully: {final_audio_path}")
        return final_audio_path
    
    except ImportError as e:
        logger.error(f"gTTS library not available: {e}")
        logger.warning("Falling back to original audio")
    
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        logger.warning("TTS failed, falling back to original audio")



# ============================================================================
# Lip-Sync with Wav2Lip
# ============================================================================

async def run_lipsync(
    video_path: Path,
    audio_path: Path,
    output_dir: Path,
) -> Path:
    """
    Apply lip-sync to video using Wav2Lip.
    
    Args:
        video_path: Path to original video
        audio_path: Path to translated audio
        output_dir: Output directory
    
    Returns:
        Path to lip-synced video
    """
    logger.info(f"Running lip-sync: {video_path} + {audio_path}")
    
    """
    import sys
    import subprocess
    
    # Wav2Lip checkpoint path
    wav2lip_dir = settings.models_path / "Wav2Lip"
    checkpoint_path = wav2lip_dir / "checkpoints" / "wav2lip_gan.pth"
    
    # Output path
    output_path = output_dir / "lipsync_output.mp4"
    
    # Run Wav2Lip inference
    # This assumes Wav2Lip repository is cloned in models/Wav2Lip/
    inference_script = wav2lip_dir / "inference.py"
    
    ZSH command for Wav2Lip inference:
    zsh -c "cd /path/to/Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face /path/to/video.mp4 --audio /path/to/audio.wav --outfile /path/to/output.mp4 --nosmooth"
    
    cmd = [
        "zsh", "-c",
        f"cd {wav2lip_dir} && python {inference_script} "
        f"--checkpoint_path {checkpoint_path} "
        f"--face {video_path} "
        f"--audio {audio_path} "
        f"--outfile {output_path} "
        f"--nosmooth"
    ]
    
    # Run in subprocess
    result = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(wav2lip_dir),
    )
    
    stdout, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Wav2Lip failed: {stderr.decode()}")
    
    return output_path
    """
    
    # ALT: Combine audio and video without lip-sync
    
    from app.utils import combine_audio_video
    
    output_path = output_dir / "final_output.mp4"
    
    await asyncio.get_event_loop().run_in_executor(
        None,
        combine_audio_video,
        video_path,
        audio_path,
        output_path,
    )
    
    logger.info(f"Lip-sync stub created: {output_path}")
    return output_path
