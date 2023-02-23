from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

# import IPython.display as ipd


models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/tts_transformer-ru-cv7_css10",
    arg_overrides={"vocoder": "hifigan", "fp16": False},
)

model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

text = "Здравствуйте, это пробный запуск."
sample = TTSHubInterface.get_model_input(task, text)
wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

###
import torchaudio as ta  # noqa

out_dir = "../out/"
# double the dimensions to get stereo audio
ta.save(f"{out_dir}/rus.wav", wav.repeat(2, 1), rate)


# change speed: sox input.wav output.wav tempy 1.33
# https://stackoverflow.com/questions/33957747/how-do-i-reduce-the-play-time-of-a-voice-mp3-file-with-sox-to-75
###
