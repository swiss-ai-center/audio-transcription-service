from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
import io
import json
from pydub import AudioSegment
from transformers import pipeline
from tempfile import NamedTemporaryFile
from fastapi import HTTPException

api_description = """
Transcribe an audio file with Whisper tiny model and return the transcription as text.
"""
api_summary = """Audio file transcription service.
"""
api_title = "Audio file transcription API."
version = "0.0.1"

settings = get_settings()
device = "cpu"


class MyService(Service):
    """
    Transcribe an audio file to text with Whisper.
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Audio Transcription",
            slug="audio-transcription",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="audio_file",
                    type=[
                        FieldDescriptionType.AUDIO_MP3,
                        FieldDescriptionType.AUDIO_OGG
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_JSON]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.SPEECH_RECOGNITION,
                    acronym=ExecutionUnitTagAcronym.SPEECH_RECOGNITION,
                ),
            ],
            has_ai=True,
            # OPTIONAL: CHANGE THE DOCS URL TO YOUR SERVICE'S DOCS
            docs_url="https://docs.swiss-ai-center.ch/reference/services/audio-transcription/",
        )
        self._logger = get_logger(settings)

        # load the model :
        self._logger.info("Loading the Whisper model...")
        self._model = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)
        self._logger.info("Whisper model loaded.")

    def process(self, data):
        # Get the audio file
        audio_data = data["audio_file"].data

        # Convert the byte data to an AudioSegment using pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))

        # Split the audio into chunks (e.g., 30 seconds each)
        chunk_duration_ms = 30 * 1000  # 30 seconds in milliseconds
        audio_chunks = [audio_segment[i:i + chunk_duration_ms] for i in range(0, len(audio_segment), chunk_duration_ms)]

        # Initialize an empty string to store the complete transcription
        complete_transcription = ""

        try:
            # Process each chunk individually
            for chunk in audio_chunks:
                with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                    # Export the chunk to a temporary WAV file
                    chunk.export(temp_audio_file.name, format="wav")

                    # Transcribe the chunk using the Whisper model
                    transcription = self._model(temp_audio_file.name)
                    # Append the transcription to the complete transcription
                    complete_transcription += transcription['text'] + " "
                    # Set complete_transcription to json
                    result = {
                        "transcription": complete_transcription,
                    }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Return the complete transcription as JSON
        return {
            "result": TaskData(
                data=json.dumps(result),
                type=FieldDescriptionType.APPLICATION_JSON
            )
        }