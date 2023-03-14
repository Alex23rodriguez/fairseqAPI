# append fairseq_repo to path
import sys

sys.path.append("./fairseq_repo")
# argparse
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument(
    "model",
    type=str,
    nargs="?",
    default="facebook/tts_transformer-ru-cv7_css10",
    help="huggingface model that will be used to transform from text to speech",
)
args = parser.parse_args()

# import necessary modules
from pathlib import Path
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    args.model,
    arg_overrides={"vocoder": "hifigan", "fp16": False},
)

# make data dir
p = Path("./data")
if not p.exists():
    p.mkdir()

q = p / str(args.model)
if not q.parent.exists():
    q.parent.mkdir()
    (p / "last_used").write_text(cfg.model.data)

# add cache location for this model
q.write_text(cfg.model.data)
