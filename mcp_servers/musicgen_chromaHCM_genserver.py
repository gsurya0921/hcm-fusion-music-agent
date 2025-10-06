import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
from mcp.server.fastmcp import FastMCP
from audiocraft.models import MusicGen
import torch
import torchaudio
import soundfile as sf
import numpy as np
import logging
import librosa
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List, Union
import json
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GEN")

mcp = FastMCP("Generation")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Load MusicGen Melody model
musicgen = MusicGen.get_pretrained('facebook/musicgen-melody', device=device)
musicgen.set_generation_params(duration=30)  # 10 seconds default
logger.info("Loaded MusicGen Melody model")

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x

def _pad_or_trim(x: torch.Tensor, T: int) -> torch.Tensor:
    t = x.size(-1)
    if t < T:
        return F.pad(x, (0, T - t))
    elif t > T:
        return x[..., :T]
    return x

def _resample(x: torch.Tensor, sr_from: int, sr_to: int) -> torch.Tensor:
    if sr_from == sr_to: 
        return x
    return torchaudio.functional.resample(x, orig_freq=sr_from, new_freq=sr_to)

def _load(path: str):
    y, sr = torchaudio.load(path)
    return y, sr

def _coerce_clip_paths(clips: Any) -> List[str]:
    """
    Accepts clips as: list[str] or list[dict] (with key 'path' or 'audio_path').
    Filters to existing files only.
    """
    out: List[str] = []
    if not clips:
        return out
    for c in clips:
        p: Optional[str] = None
        if isinstance(c, str):
            p = c
        elif isinstance(c, dict):
            p = c.get("path") or c.get("audio_path") or c.get("file")
        if p and os.path.isfile(p):
            out.append(p)
    return out


def combine_raag_clips_gpu(clips: List[Any], target_sr: int = 32000, 
                           target_duration: int = 10, device: str = None,
                           num_workers: int = 4) -> torch.Tensor:
    """
    Load multiple raag clips, resample, and combine with weighted average
    Returns: [1, T] tensor at target_sr
    """
    if not clips:
        raise ValueError("No clips provided")
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clips = _coerce_clip_paths(clips)
    
    if not clips:
        raise ValueError("No valid clip paths found")
    
    logger.info(f"Loading {len(clips)} raag clips...")
    
    # Load all clips in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        loaded = list(ex.map(_load, clips))
    
    # Target length in samples
    T = target_sr * target_duration
    
    # Process each clip
    proc = []
    with torch.inference_mode():
        for wav, sr in loaded:
            # Convert to mono
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            wav = wav.to(device, non_blocking=True)
            wav = _resample(wav, sr_from=sr, sr_to=target_sr)
            wav = _pad_or_trim(wav, T)
            proc.append(wav)
        
        # Stack and average
        stack = torch.stack(proc, dim=0)  # [N, 1, T]
        combined = stack.mean(dim=0)      # [1, T]
        
        logger.info(f"Combined {len(clips)} clips into shape {combined.shape}")
    
    return combined


# Thread pool for blocking GPU operations
gpu_executor = ThreadPoolExecutor(max_workers=1)

@mcp.tool()
async def generate_layer1_music(
    prompt: str,
    out_dir: Optional[str] = None,
    seconds: float = 10.0,
    sample_rate: int = 32000,
    seed: Optional[int] = None,
) -> str:
    """
    Generate Layer 1 music using base MusicGen (no melody conditioning).
    This is for when you just want to generate from text prompt.
    """
    def _do_generation():
        musicgen.set_generation_params(duration=seconds)
        
        outputs = musicgen.generate([prompt], progress=True)
        outputs = outputs[0].cpu().numpy().squeeze()
        
        path = '/home/ganesh/fubar/music_agent_v1/layer1.wav'
        sf.write(path, outputs, sample_rate)
        logger.info(f"Generated Layer 1 at: {path}")
        
        # Cleanup
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return path
    
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(gpu_executor, _do_generation)
    
    result = json.dumps({"output_path": path})
    return result


@mcp.tool()
async def generate_layer2_music(
    layer1_audio_path: str,
    prompt: str,
    knob: float = 0.5,
    instrument: Optional[str] = None,
    tempo: Optional[int] = None,
    style: Optional[str] = None,
    out_dir: Optional[str] = None,
    seconds: float = 30.0,
    sample_rate: int = 32000,
    seed: Optional[int] = None,
) -> str:
    """
    Generate Layer 2 music using MusicGen Melody.
    Uses layer1_audio as melody conditioning.
    
    Args:
        layer1_audio_path: Path to Layer 1 raag foundation audio
        prompt: Text description for Layer 2 generation
        knob: Conditioning strength (0-1). Higher = stronger melody adherence
        instrument: Optional instrument name (e.g., "flute")
        tempo: Optional tempo in BPM
        style: Optional style (e.g., "rock", "jazz")
    """
    def _do_generation():
        # Build enhanced prompt
        enhanced_prompt = prompt
        if instrument:
            enhanced_prompt = f"{instrument} melody, {enhanced_prompt}"
        if tempo:
            enhanced_prompt = f"{enhanced_prompt}, {tempo} BPM"
        if style:
            enhanced_prompt = f"{enhanced_prompt}, {style} style"
        
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        logger.info(f"Melody conditioning from: {layer1_audio_path}")
        
        # Load melody audio
        melody_wav, melody_sr = torchaudio.load(layer1_audio_path)
        
        # Convert to mono if stereo
        if melody_wav.shape[0] > 1:
            melody_wav = melody_wav.mean(dim=0, keepdim=True)
        
        # Resample to 32kHz if needed
        if melody_sr != 32000:
            melody_wav = torchaudio.functional.resample(
                melody_wav, 
                orig_freq=melody_sr, 
                new_freq=32000
            )
        
        logger.info(f"Melody shape: {melody_wav.shape}, sr: 32000")
        
        # Set generation parameters
        # cfg_coef controls how much the model follows the text vs melody
        # Higher = stronger text adherence, lower = stronger melody adherence
        cfg_coef = 3.0 + (knob * 7.0)  # Range: 3.0 to 10.0
        
        musicgen.set_generation_params(
            duration=seconds,
            temperature=1.0,
            top_k=250,
            top_p=0.0,
            cfg_coef=cfg_coef
        )
        
        logger.info(f"Generating with cfg_coef={cfg_coef:.1f}")
        
        # Generate with melody conditioning
        outputs = musicgen.generate_with_chroma(
            descriptions=[enhanced_prompt],
            melody_wavs=melody_wav,
            melody_sample_rate=32000,
            progress=True
        )
        
        # Ensure CUDA sync before CPU operations
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        outputs_cpu = outputs[0].cpu().numpy().squeeze()
        
        path = '/home/ganesh/fubar/music_agent_v1/generated_layer2.wav'
        sf.write(path, outputs_cpu, sample_rate)
        logger.info(f"Generated Layer 2 at: {path}")
        
        # ADD THIS - force flush stdout
        # import sys
        # sys.stdout.flush()
        # sys.stderr.flush()
        # Cleanup
        del outputs, melody_wav, outputs_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return path
    
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(gpu_executor, _do_generation)
    
    result = json.dumps({"output_path": path})
    logger.info(f"Returning result: {result}")
    return result


@mcp.tool()
async def generate_fusion_from_raag_clips(
    raag_clips: List[Any],
    prompt: str,
    knob: float = 0.6,
    instrument: Optional[str] = None,
    tempo: Optional[int] = None,
    style: Optional[str] = None,
    seconds: float = 10.0,
    sample_rate: int = 32000,
) -> str:
    """
    All-in-one: Combine raag clips and generate fusion music.
    
    Args:
        raag_clips: List of audio file paths or dicts with 'audio_path' key
        prompt: Base text prompt
        knob: Fusion intensity (0=pure raag, 1=full transformation)
        instrument: Instrument for Layer 2 (e.g., "flute", "sitar")
        tempo: Tempo in BPM
        style: Musical style (e.g., "rock", "jazz", "electronic")
    """
    def _do_generation():
        # Step 1: Combine raag clips
        logger.info("Combining raag clips...")
        combined_melody = combine_raag_clips_gpu(
            raag_clips, 
            target_sr=32000,
            target_duration=10,
            device=device
        )
        
        # Step 2: Build prompt
        enhanced_prompt = prompt
        if instrument:
            enhanced_prompt = f"{instrument} melody, {enhanced_prompt}"
        if tempo:
            enhanced_prompt = f"{enhanced_prompt}, {tempo} BPM"
        if style:
            enhanced_prompt = f"{enhanced_prompt}, {style} style"
        
        logger.info(f"Generating fusion with prompt: {enhanced_prompt}")
        
        # Step 3: Generate with melody conditioning
        cfg_coef = 3.0 + (knob * 7.0)
        musicgen.set_generation_params(
            duration=seconds,
            cfg_coef=cfg_coef,
            temperature=1.0
        )
        
        outputs = musicgen.generate_with_chroma(
            descriptions=[enhanced_prompt],
            melody_wavs=combined_melody,
            melody_sample_rate=32000,
            progress=True
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        outputs_cpu = outputs[0].cpu().numpy().squeeze()
        
        path = '/home/ganesh/fubar/music_agent_v1/generated_fusion.wav'
        sf.write(path, outputs_cpu, sample_rate)
        logger.info(f"Generated fusion music at: {path}")
        
        # Cleanup
        del outputs, combined_melody, outputs_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return path
    
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(gpu_executor, _do_generation)
    
    result = json.dumps({"output_path": path})
    return result


if __name__ == "__main__":
    logger.info("Starting MCP GEN server with MusicGen Melody")
    mcp.run(transport="stdio")