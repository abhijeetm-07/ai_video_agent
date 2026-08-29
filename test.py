from utils.audio_processor import process_input
from core.transcriber import transcribe_all

source = "https://www.youtube.com/watch?v=Lc_HY4uv3K8"
language = "hinglish"
chunks = process_input(source)
transcription = transcribe_all(chunks,language=language)
print("Full Transcription:\n")
print(transcription)