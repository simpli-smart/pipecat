from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType
from pipecat.frames.frames import Frame, InputAudioRawFrame, StartFrame, TextFrame, OutputAudioRawFrame, TransportMessageUrgentFrame
import base64

class AudioWebsocketSerializer(FrameSerializer):
    def __init__(self):
        self.sample_rate = 16000
        self.num_channels = 1

    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.BINARY
    
    async def setup(self, frame: StartFrame):
        self.sample_rate = frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> bytes:
        return frame.audio if isinstance(frame, InputAudioRawFrame) else None

    async def deserialize(self, data: bytes) -> Frame:
        return InputAudioRawFrame(audio=data, sample_rate=self.sample_rate, num_channels=self.num_channels)

class OpenAIRealTimeWebsocketSerializer(FrameSerializer):
    def __init__(self):
        self.sample_rate = 16000
        self.num_channels = 1

    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.BINARY
    
    async def setup(self, frame: StartFrame):
        self.sample_rate = frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> dict:
        if isinstance(frame, OutputAudioRawFrame):
            return {"type": "response.output_audio.delta", "delta": base64.b64encode(frame.audio).decode("utf-8")}
        elif isinstance(frame, TransportMessageUrgentFrame):
            msg_type = frame.message["type"]
            if msg_type == "bot-llm-stopped":
                return {"type": "response.output_text.done"}
            elif msg_type == "user-transcription":
                return {"type": "response.output_audio_transcript.delta", "delta": frame.message["data"]["text"]}
            elif msg_type == "bot-llm-text":
                return {"type": "response.output_text.delta", "delta": frame.message["data"]["text"]}
            elif msg_type == "bot-stopped-speaking":
                return {"type": "response.done"}
        else:
            return None

    async def deserialize(self, data: bytes | dict) -> Frame:
        if isinstance(data, dict):
            event_type = data["type"]
            if event_type == "input_audio_buffer.append":
                data = base64.b64decode(data["audio"])
                return InputAudioRawFrame(audio=data, sample_rate=self.sample_rate, num_channels=self.num_channels)
            
        elif isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=self.sample_rate, num_channels=self.num_channels)
        else:
            return None