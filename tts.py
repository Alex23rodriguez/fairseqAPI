# append fairseq_repo to path
from pathlib import Path
import sys
from argparse import ArgumentParser

sys.path.append("./fairseq_repo")


parser = ArgumentParser(
    prog="Russian Text To Speech",
    description="given some russian text, generates a .wav file of spoken russian",
)

parser.add_argument("-o", "--outfile", default="./out/out.wav")
parser.add_argument("text", nargs="+")
parser.add_argument("-m", "--model", default=None)

args = parser.parse_args()


# import necessary modules

import torchaudio as ta
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

# load model
if args.model:
    data_path = Path(f"./data/{args.model}")
else:
    data_path = Path(f"./data/last_used")
data = data_path.read_text()
filename = data + "/model.pt"

arg_overrides = {"vocoder": "hifigan", "fp16": False, "data": data}
models, cfg, task = load_model_ensemble_and_task(
    [str(filename)], arg_overrides=arg_overrides
)


model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)


def tts(text, file):
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    ta.save(file, wav.repeat(2, 1), rate)


### convert to audio
p = Path(args.outfile)
if not p.parent.exists():
    p.parent.mkdir()
tts(" ".join(args.text), args.outfile)
