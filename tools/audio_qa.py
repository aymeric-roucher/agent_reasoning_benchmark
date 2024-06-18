from transformers.agents import Tool, HfEngine, SpeechToTextTool
from transformers.pipelines.audio_utils import ffmpeg_read


class AudioQATool(Tool):
    name = "audio_qa"
    description = "A tool to answer questions about audio files. Handles [.wav, .mp3, .flac] extensions."
    inputs = {
        "question": {"description": "the question to answer", "type": "text"},
        "audio_path": {
            "description": "the path to the audio",
            "type": "text",
        },
    }
    output_type = "text"

    def __init__(
            self,
            llm_engine,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.stt = SpeechToTextTool(**kwargs)
        self.engine = llm_engine

    def forward(self, question: str, audio_path: str) -> str:
        # try:
        audio = open(audio_path, "rb").read()
        inputs = ffmpeg_read(audio, 16000)
        transcript = self.stt(inputs)
        # except Exception as e:
        #     raise RuntimeError(str(e))
        messages = [
            {
                "role": "user",
                "content": "You need to answer the following question with the below context: "
                + question,
            },
            {
                "role": "user",
                "content": "Here is the context:\n" + transcript,
            },
        ]

        return self.engine(messages)

