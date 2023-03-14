# fairseqAPI

## Small example use of the fairseq python library

This is a proof of concept of how the [fairseq](https://github.com/facebookresearch/fairseq) python library can be used for `tts`, given that the examples on HuggingFace and the repo are either unclear, or broken.

# Installation
(This installation should work for python versions `3.6` through `3.10`. version `3.11` is not yet supported. Run `python --version` to view current installation)
- clone this repository to your local machine and cd into it
```bash
git clone https://github.com/Alex23rodriguez/fairseqAPI.git
cd fairseqAPI
```
- clone facebook's fairseq repository, as there are many modules that we use. Notice that we change the name to avoid confision between the module and the repo folder.
```bash
git clone --depth 1 https://github.com/pytorch/fairseq fairseq_repo
```
NOTE: this is different from `pip install fairseq`, which installs the CLI tool, _not_ the python module. Check out their repo for more info.
- install the required modules
  - it is recommended to create a virtual envirenment before installing requirements: `python -m venv venv && source venv/bin/activate`
```bash
pip install -r requirements.txt
```

# Usage
## downloading models
Once the setup is done, we can move on to downloading a model from Hugging Face. Given that I personally use this for Russian, the model defaults to [this one](https://huggingface.co/facebook/tts_transformer-ru-cv7_css10) (note that the example on HG's website is currently broken)
- to download the russian tts model, simply run the following command:
```bash
python download_hf_model.py
```
- to download any other model, pass it as a commandline argument
```bash
python download_hf_model.py facebook/tts_transformer-ru-cv7_css10
```
- note that models are typically heavy, which is why we only want to do this step once

## using models
To use `tts`, simply call `tts.py` with the corresponding commandline arguments (`python tts.py -h` for more info).
```bash
python tts.py Здравствуйте, это пробный запуск. До свидание.

```
- output defaults to `./out/out.wav`
  - to play the output from the terminal install `sox` and use `play out/out.wav`
  - `sox` can also be used to change the speed of the recording: `sox input.wav output.wav tempo 1.33`
