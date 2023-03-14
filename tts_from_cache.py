# import os
import sys
from argparse import ArgumentParser, FileType

parser = ArgumentParser(
    prog="Russian Text To Speech",
    description="given some russian text, generates a .wav file of spoken russian",
)

parser.add_argument("-o", "--outfile", default="./out/out.wav")
parser.add_argument("text", nargs="+")
args = parser.parse_args()

sys.path.append("./fairseq_repo")

import torchaudio as ta
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

data = "/Users/alex/.cache/fairseq/models--facebook--tts_transformer-ru-cv7_css10/snapshots/9eaa70aaff5aac14f8e17856b3def855ebdbf69c"  # noqa
filename = data + "/model.pt"
arg_overrides = {"vocoder": "hifigan", "fp16": False, "data": data}
models, cfg, task = load_model_ensemble_and_task(
    [filename], arg_overrides=arg_overrides
)


model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)


def tts(text, file):
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    ta.save(file, wav.repeat(2, 1), rate)


### convert to audio
tts(" ".join(args.text), args.outfile)
