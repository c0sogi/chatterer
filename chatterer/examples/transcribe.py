# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportMissingTypeStubs=false

from io import BytesIO
from pathlib import Path
from typing import List, Optional

import typer
from openai import OpenAI
from pydub import AudioSegment


# -------------------------------------------------------------------
# Helper functions for timestamp parsing & segment selection
# -------------------------------------------------------------------
def parse_timestamp(ts: str) -> float:
    """
    Parse a timestamp string into seconds.
    Supports:
      - "SS" or "SS.sss"
      - "MM:SS" or "MM:SS.sss"
      - "HH:MM:SS" or "HH:MM:SS.sss"
    """
    parts = ts.split(":")
    seconds = 0.0
    for idx, part in enumerate(reversed(parts)):
        if not part:
            value = 0.0
        else:
            value = float(part)
        if idx == 0:
            seconds += value
        elif idx == 1:
            seconds += value * 60
        elif idx == 2:
            seconds += value * 3600
        else:
            raise ValueError(f"Timestamp '{ts}' is too long (use H:MM:SS at most)")
    return seconds


def get_selected_audio(audio: AudioSegment, segments_str: str) -> AudioSegment:
    """
    Given full audio and a segments string (e.g. "650-750,16:50-17:30,800-"),
    extract those subranges and concatenate them.
    """
    duration_ms = len(audio)
    duration_s = duration_ms / 1000.0
    subsegments: List[AudioSegment] = []

    for part in segments_str.split(","):
        if "-" not in part:
            raise ValueError(f"Invalid segment '{part}' (must contain '-')")
        start_str, end_str = part.split("-", 1)
        start_s = parse_timestamp(start_str) if start_str.strip() else 0.0
        end_s = parse_timestamp(end_str) if end_str.strip() else duration_s

        # clamp
        start_s = max(0.0, min(start_s, duration_s))
        end_s = max(0.0, min(end_s, duration_s))
        if end_s <= start_s:
            print(f"[!] Warning: segment '{part}' yields non-positive duration; skipping.")
            continue

        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        sub = audio[start_ms:end_ms]
        subsegments.append(sub)
        print(f"[i] Selected segment {start_s:.2f}s\u2013{end_s:.2f}s ({end_s - start_s:.2f}s)")

    if not subsegments:
        raise RuntimeError("No valid segments were specified.")
    # concatenate
    combined = subsegments[0]
    for seg in subsegments[1:]:
        combined += seg
    return combined


# -------------------------------------------------------------------
# Main transcription logic
# -------------------------------------------------------------------
def command(
    audio_path: Path = typer.Argument(help="The audio file to transcribe."),
    output: Optional[Path] = typer.Option(None, help="Path to save the transcription output."),
    model: str = typer.Option("gpt-4o-transcribe", help="The model to use for transcription."),
    api_key: Optional[str] = typer.Option(None, help="The API key for authentication."),
    base_url: str = typer.Option("https://api.openai.com/v1", help="The base URL for the API."),
    prompt: str = typer.Option("Transcribe whole text from audio.", help="The prompt to use for transcription."),
    segments: Optional[str] = typer.Option(None, help="Comma-separated list of time ranges to include (e.g. '650-750,16:50-17:30,800-')."),
    max_chunk_duration: int = typer.Option(600, help="Maximum duration of each chunk in seconds."),
) -> None:
    """Transcribe audio files to text."""
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 1) Load entire audio
    original_audio = load_audio_segment(audio_path)

    # 2) If segments specified, extract & combine
    if segments:
        audio = get_selected_audio(original_audio, segments)
        print(f"[i] Combined audio duration: {len(audio) / 1000:.1f}s (from segments)")
    else:
        audio = original_audio
        print(f"[i] Audio duration: {len(audio) / 1000:.1f}s (full audio)")

    # 3) Split into chunks
    audio_segments = split_audio(audio, max_chunk_duration)
    print(f"[i] Splitting into {len(audio_segments)} segment(s) for transcription")

    # 4) Transcribe each chunk
    transcripts: List[str] = []
    for idx, seg in enumerate(audio_segments, start=1):
        print(f"[i] Transcribing segment {idx}/{len(audio_segments)}...")
        transcripts.append(transcribe_segment(seg, client, model, prompt))

    # 5) Write out
    full = "\n\n".join(transcripts)
    out_path = output or audio_path.with_suffix(".txt")
    out_path.write_text(full, encoding="utf-8")
    print(f"[+] Transcription saved to: {out_path}")


def load_audio_segment(file_path: Path) -> AudioSegment:
    """
    Load an audio file as an AudioSegment. Convert to mp3 format in-memory if needed.
    """
    ext = file_path.suffix.lower()[1:]
    audio = AudioSegment.from_file(file_path.as_posix(), format=None if ext == "mp3" else ext)
    if ext != "mp3":
        buffer = BytesIO()
        audio.export(buffer, format="mp3")
        buffer.seek(0)
        audio = AudioSegment.from_file(buffer, format="mp3")
    return audio


def split_audio(audio: AudioSegment, max_duration_s: int) -> List[AudioSegment]:
    """
    Split the AudioSegment into chunks no longer than max_duration_s seconds.
    """
    chunk_ms = (max_duration_s - 1) * 1000
    duration_ms = len(audio)
    segments: List[AudioSegment] = []
    for start in range(0, duration_ms, chunk_ms):
        end = min(start + chunk_ms, duration_ms)
        segments.append(audio[start:end])
    return segments


def transcribe_segment(segment: AudioSegment, client: OpenAI, model: str, prompt: str) -> str:
    """
    Transcribe a single AudioSegment chunk and return its text.
    """
    buffer = BytesIO()
    segment.export(buffer, format="mp3")
    buffer.seek(0)
    mp3_bytes = buffer.read()

    response = client.audio.transcriptions.create(
        model=model,
        prompt=prompt,
        file=("audio.mp3", mp3_bytes),
        response_format="text",
        stream=True,
    )
    for res in response:
        if res.type == "transcript.text.delta":
            print(res.delta, end="", flush=True)
        elif res.type == "transcript.text.done":
            print()
            return res.text
    raise RuntimeError("No transcription result found.")
