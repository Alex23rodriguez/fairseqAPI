import pickle
import torchaudio as ta
from fairseq import tasks
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

with open("../pickles/models.p", "rb") as f:
    models = pickle.load(f)

with open("../pickles/cfg.p", "rb") as f:
    cfg = pickle.load(f)

### get the task

data = "/Users/alex/.cache/fairseq/models--facebook--tts_transformer-ru-cv7_css10/snapshots/9eaa70aaff5aac14f8e17856b3def855ebdbf69c"  # noqa
filename = data + "/model.pt"
arg_overrides = {"vocoder": "hifigan", "fp16": False, "data": data}
state = load_checkpoint_to_cpu(filename, arg_overrides)
task = tasks.setup_task(cfg.task, from_checkpoint=True)
if "task_state" in state:
    task.load_state_dict(state["task_state"])

### prepare data

model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

### tts function


def tts(text, file):
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    ta.save(file, wav.repeat(2, 1), rate)


### convert to audio

text = """
Можешь молчать, я не обижен
Убегать, но я о себе дам знать, пока я чувствую и вижу
"""
outfile = "../out/mytest2.wav"
tts(text, outfile)
