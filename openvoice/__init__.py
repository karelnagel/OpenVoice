import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from openai import OpenAI
from dotenv import load_dotenv
import subprocess

load_dotenv()


def tts(text: str, out_path: str) -> str:
    client = OpenAI()
    res = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )
    res.stream_to_file(out_path)
    return out_path


SPEAKER_EMBEDDING_CREATION_TEXT = (
    "This audio will be used to extract the base speaker tone color embedding. "
    + "Typically a very short audio should be sufficient, but increasing the audio "
    + "length will also improve the output audio quality."
)


class OpenVoice:
    def __init__(self):
        self.ckpt_path = os.path.expanduser("~/.cache/openvoice")
        convert_path = f"{self.ckpt_path}/checkpoints/converter"
        checkpoint_path = f"{convert_path}/checkpoint.pth"
        config_path = f"{convert_path}/config.json"
        if not os.path.exists(checkpoint_path) or not os.path.exists(config_path):
            print("Downloading checkpoints...")
            os.makedirs(self.ckpt_path, exist_ok=True)
            checkpoint_url = "https://myshell-public-repo-hosting.s3.amazonaws.com/checkpoints_1226.zip"
            subprocess.run(
                f"wget {checkpoint_url} -O {self.ckpt_path}/checkpoints_1226.zip",
                shell=True,
                check=True,
            )
            subprocess.run(
                f"unzip {self.ckpt_path}/checkpoints_1226.zip -d {self.ckpt_path}",
                shell=True,
                check=True,
            )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ToneColorConverter(config_path, device=self.device)
        self.model.load_ckpt(checkpoint_path)
        self.src_se = self._get_src_embedding()

    def _get_src_embedding(self) -> torch.Tensor:
        src_se_path = f"{self.ckpt_path}/src_se.pth"
        if os.path.exists(src_se_path):
            src_se = torch.load(src_se_path)
        else:
            print("Generating src speaker embedding...")
            tts_path = tts(
                SPEAKER_EMBEDDING_CREATION_TEXT, f"{self.ckpt_path}/src_tts.mp3"
            )
            src_se = self.create_speaker_embedding(tts_path)
            torch.save(src_se, src_se_path)
        return src_se

    def create_speaker_embedding(self, file_path: str) -> torch.Tensor:
        se, audio_name = se_extractor.get_se(file_path, self.model, vad=True)
        return se

    def infer(self, text: str, out_path: str, target_se: torch.Tensor):
        tts_path = tts(text, f"{out_path}_tts.mp3")
        self.model.convert(
            audio_src_path=tts_path,
            src_se=self.src_se,
            tgt_se=target_se,
            message="default",
            output_path=out_path,
        )
        return out_path
