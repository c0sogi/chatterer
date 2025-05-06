# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportMissingTypeStubs=false

from io import BytesIO
from pathlib import Path
from typing import cast

from openai import OpenAI
from pydub import AudioSegment
from spargear import ArgumentSpec, BaseArguments

# Maximum chunk length in seconds
MAX_CHUNK_DURATION = 600


class TranscriptionApiArguments(BaseArguments):
    in_path = ArgumentSpec(
        ["in-path"],
        type=Path,
        help="The audio file to transcribe.",
    )
    out_path = ArgumentSpec(
        ["--out-path"],
        type=Path,
        default=None,
        help="Path to save the transcription output.",
    )
    model: ArgumentSpec[str] = ArgumentSpec(
        ["--model"],
        default="gpt-4o-transcribe",
        help="The model to use for transcription.",
    )
    api_key: ArgumentSpec[str] = ArgumentSpec(
        ["--api-key"],
        default=None,
        help="The API key for authentication.",
    )
    base_url: ArgumentSpec[str] = ArgumentSpec(
        ["--base-url"],
        default="https://api.openai.com/v1",
        help="The base URL for the API.",
    )

    def run(self) -> None:
        audio_path = self.in_path.unwrap()
        model = self.model.unwrap()

        client = OpenAI(api_key=self.api_key.value, base_url=self.base_url.value)

        audio = load_audio_segment(audio_path)

        segments = split_audio(audio, MAX_CHUNK_DURATION)
        print(f"[i] Audio duration: {len(audio) / 1000:.1f}s; splitting into {len(segments)} segment(s)")

        transcripts: list[str] = []
        for idx, seg in enumerate(segments, start=1):
            print(f"[i] Transcribing segment {idx}/{len(segments)}...")
            transcripts.append(transcribe_segment(seg, client, model))

        full_transcript = "\n\n".join(transcripts)
        output_path: Path = self.out_path.value or audio_path.with_suffix(".txt")
        output_path.write_text(full_transcript, encoding="utf-8")
        print(f"[✓] Transcription saved to: {output_path}")


def load_audio_segment(file_path: Path) -> AudioSegment:
    """
    Load an audio file as an AudioSegment. Convert to mp3 format in-memory if needed.
    """
    ext = file_path.suffix.lower()[1:]
    audio = AudioSegment.from_file(file_path.as_posix(), format=ext if ext != "mp3" else None)
    if ext != "mp3":
        buffer = BytesIO()
        audio.export(buffer, format="mp3")
        buffer.seek(0)
        audio = AudioSegment.from_file(buffer, format="mp3")
    return audio


def split_audio(audio: AudioSegment, max_duration_s: int) -> list[AudioSegment]:
    """
    Split the AudioSegment into chunks no longer than max_duration_s seconds.
    """
    chunk_length_ms = (max_duration_s - 1) * 1000
    duration_ms = len(audio)
    segments: list[AudioSegment] = []
    segment_idx: int = 0
    for start_ms in range(0, duration_ms, chunk_length_ms):
        end_ms = min(start_ms + chunk_length_ms, duration_ms)
        segment = cast(AudioSegment, audio[start_ms:end_ms])
        segments.append(segment)
        # with open(f"segment_{segment_idx}.mp3", "wb") as f:
        #     segment.export(f, format="mp3")
        segment_idx += 1
    return segments


def transcribe_segment(segment: AudioSegment, client: OpenAI, model: str) -> str:
    """
    Transcribe a single AudioSegment chunk and return its text.
    """
    buffer = BytesIO()
    segment.export(buffer, format="mp3")
    buffer.seek(0)
    mp3_bytes = buffer.read()
    response = client.audio.transcriptions.create(
        model=model,
        prompt="Transcribe whole text from audio.",
        file=("audio.mp3", mp3_bytes),
        response_format="text",
        stream=True,
    )
    for res in response:
        if res.type == "transcript.text.delta":
            print(res.delta, end="", flush=True)
        if res.type == "transcript.text.done":
            print()
            return res.text
    else:
        raise RuntimeError("No transcription result found.")


if __name__ == "__main__":
    TranscriptionApiArguments().run()
