<!-- start installation -->
## Installation
🐸coqui-ai-TTS-rocm is tested on Arch Linux with python == 3.14.

# prerequisites:
```
sudo pacman -Sy wget git python python-torchcodec python-pytorch-opt-rocm fakeroot debugedit cmake base-devel python-pip --noconfirm
git clone https://aur.archlinux.org/python-torchaudio-rocm.git
cd python-torchaudio-rocm
makepkg -si
cd ..
git clone https://github.com/Zahnschmelz/coqui-ai-TTS-rocm.git
cd coqui-ai-TTS-rocm
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .
pip install transformers==5.0.0
```
# check if pytorch working:
```
python -c "import torch; torch.cuda.is_available()"
```
# download a moedel (for example xtts-v2.0.3)
```
mkdir -p models/xtts-v2.0.3 && cd models/xtts-v2.0.3
wget https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/.gitattributes?download=true
wget https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/LICENSE.txt?download=true
wget https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/README.md?download=true
wget https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/config.json?download=true
wget https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/dvae.pth?download=true
wget https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/hash.md5?download=true
wget https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/mel_stats.pth?download=true
wget https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/model.pth?download=true
wget https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/speakers_xtts.pth?download=true
wget https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/vocab.json?download=true
```
> [!NOTE]
> From `coqui-tts` 0.27.4, PyTorch is not included by default and you need to install it yourself.

🐸TTS is tested on Ubuntu 24.04 with **python >= 3.10, < 3.15** and PyTorch
2.2+, but should also work on Mac and Windows.

It is strongly recommended to use [uv](https://docs.astral.sh/uv/) to install
everything into a virtual environment (otherwise leave out `uv` from the
commands below).

First install PyTorch, `torchaudio`, and (only for PyTorch 2.9+) `torchcodec`
with their [official instructions](https://pytorch.org/get-started/locally/),
choosing the CPU/CUDA/ROCm version as necessary. Or let uv automatically select
the right version for your system:

```bash
uv pip install torch torchaudio torchcodec --torch-backend=auto
```

If you are only interested in [synthesizing speech](https://coqui-tts.readthedocs.io/en/latest/inference.html) with the pretrained 🐸TTS models, installing from PyPI is the easiest option.

```bash
uv pip install coqui-tts
```

If you plan to code or train models, clone 🐸TTS and install it locally.

```bash
git clone https://github.com/idiap/coqui-ai-TTS
cd coqui-ai-TTS
uv pip install -e .
```

### Optional dependencies

The following extras allow the installation of optional dependencies:

| Name | Description |
|------|-------------|
| `all` | All optional dependencies |
| `notebooks` | Dependencies only used in notebooks |
| `server` | Dependencies to run the TTS server |
| `bn` | Bangla G2P |
| `ja` | Japanese G2P |
| `ko` | Korean G2P |
| `zh` | Chinese G2P |
| `languages` | All language-specific dependencies |

You can install extras with one of the following commands:

```bash
uv pip install coqui-tts[server,ja]
uv pip install -e .[server,ja]
```

### Pytorch extras

There are also the following convenience extras to automatically install the
PyTorch dependencies. Note that the CPU/CUDA selection only works with uv and
when installing Coqui from source. With other package managers or when installing
`coqui-tts` from PyPI, the PyTorch dependencies will be installed from PyPI.

| Name | Description |
|------|-------------|
| `cpu` | Install `torch`, `torchaudio` (CPU) |
| `cuda` | Install `torch`, `torchaudio` (CUDA) |
| `codec` | Install `torchcodec` (CPU), needed with PyTorch>=2.9 |
| `codec-cuda` | Install `torchcodec` (CUDA), needed with PyTorch>=2.9 |

### Platforms

If you are on Ubuntu (Debian), you can also run the following commands for installation.

```bash
make system-deps
make install
```

<!-- end installation -->

## Docker Image
You can also try out Coqui TTS without installation with the docker image.
Simply run the following command and you will be able to run TTS:

```bash
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/idiap/coqui-tts-cpu
python3 TTS/server/server.py --list_models #To get the list of available models
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits # To start a server
```

You can then enjoy the TTS server [here](http://localhost:5002/). More details,
like GPU support and a Docker Compose configuration, can be found [in the
documentation](https://coqui-tts.readthedocs.io/en/latest/docker_images.html).


## Synthesizing speech by 🐸TTS
<!-- start inference -->
### 🐍 Python API

#### Multi-speaker and multi-lingual model

```python
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Initialize TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# List speakers
print(tts.speakers)

# Run TTS
# ❗ XTTS supports both, but many models allow only one of the `speaker` and
# `speaker_wav` arguments

# TTS with list of amplitude values as output, clone the voice from `speaker_wav`
wav = tts.tts(
  text="Hello world!",
  speaker_wav="my/cloning/audio.wav",
  language="en"
)

# TTS to a file, use a preset speaker
tts.tts_to_file(
  text="Hello world!",
  speaker="Craig Gutsy",
  language="en",
  file_path="output.wav"
)
```

From version 0.27.0 you can [cache cloned
voices](https://coqui-tts.readthedocs.io/en/latest/cloning.html) with a custom
`speaker` ID, so you only need to pass audio files in `speaker_wav` once.

> [!NOTE]
> For more control or additional outputs, e.g. timestamps, use the lower-level
> [Synthesizer API](https://coqui-tts.readthedocs.io/en/latest/main_classes/synthesizer.html).

#### Single speaker model

```python
# Initialize TTS with the target model name
tts = TTS("tts_models/de/thorsten/tacotron2-DDC").to(device)

# Run TTS
tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path=OUTPUT_PATH)
```

#### Voice conversion (VC)

Converting the voice in `source_wav` to the voice of `target_wav`:

```python
tts = TTS("voice_conversion_models/multilingual/vctk/freevc24").to("cuda")
tts.voice_conversion_to_file(
  source_wav="my/source.wav",
  target_wav="my/target.wav",
  file_path="output.wav"
)
```

Other available voice conversion models:
- `voice_conversion_models/multilingual/multi-dataset/knnvc`
- `voice_conversion_models/multilingual/multi-dataset/openvoice_v1`
- `voice_conversion_models/multilingual/multi-dataset/openvoice_v2`

For more details, see this
[dedicated page](https://coqui-tts.readthedocs.io/en/latest/vc.html).

#### Voice cloning by combining single speaker TTS model with the default VC model

This way, you can clone voices by using any model in 🐸TTS. The FreeVC model is
used for voice conversion after synthesizing speech.

```python

tts = TTS("tts_models/de/thorsten/tacotron2-DDC")
tts.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)
```

#### TTS using Fairseq models in ~1100 languages 🤯
For Fairseq models, use the following name format: `tts_models/<lang-iso_code>/fairseq/vits`.
You can find the language ISO codes [here](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html)
and learn about the Fairseq models [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms).

```python
# TTS with fairseq models
api = TTS("tts_models/deu/fairseq/vits")
api.tts_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    file_path="output.wav"
)
```

**Note:** Some Fairseq models need the romanization library `uroman` to be
installed. For this you can install `coqui-tts` with the `languages` extra.

### Command-line interface `tts`

<!-- begin-tts-readme -->

Synthesize speech on the command line.

You can either use your trained model or choose a model from the provided list.

- List provided models:

  ```sh
  tts --list_models
  ```

- Get model information. Use the names obtained from `--list_models`.
  ```sh
  tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
  ```
  For example:
  ```sh
  tts --model_info_by_name tts_models/tr/common-voice/glow-tts
  tts --model_info_by_name vocoder_models/en/ljspeech/hifigan_v2
  ```

#### Single speaker models

- Run TTS with the default model (`tts_models/en/ljspeech/tacotron2-DDC`):

  ```sh
  tts --text "Text for TTS" --out_path output/path/speech.wav
  ```

- Run TTS and pipe out the generated TTS wav file data:

  ```sh
  tts --text "Text for TTS" --pipe_out --out_path output/path/speech.wav | aplay
  ```

- Run a TTS model with its default vocoder model:

  ```sh
  tts --text "Text for TTS" \
      --model_name "<model_type>/<language>/<dataset>/<model_name>" \
      --out_path output/path/speech.wav
  ```

  For example:

  ```sh
  tts --text "Text for TTS" \
      --model_name "tts_models/en/ljspeech/glow-tts" \
      --out_path output/path/speech.wav
  ```

- Run with specific TTS and vocoder models from the list. Note that not every vocoder is compatible with every TTS model.

  ```sh
  tts --text "Text for TTS" \
      --model_name "<model_type>/<language>/<dataset>/<model_name>" \
      --vocoder_name "<model_type>/<language>/<dataset>/<model_name>" \
      --out_path output/path/speech.wav
  ```

  For example:

  ```sh
  tts --text "Text for TTS" \
      --model_name "tts_models/en/ljspeech/glow-tts" \
      --vocoder_name "vocoder_models/en/ljspeech/univnet" \
      --out_path output/path/speech.wav
  ```

- Run your own TTS model (using Griffin-Lim Vocoder):

  ```sh
  tts --text "Text for TTS" \
      --model_path path/to/model.pth \
      --config_path path/to/config.json \
      --out_path output/path/speech.wav
  ```

- Run your own TTS and Vocoder models:

  ```sh
  tts --text "Text for TTS" \
      --model_path path/to/model.pth \
      --config_path path/to/config.json \
      --out_path output/path/speech.wav \
      --vocoder_path path/to/vocoder.pth \
      --vocoder_config_path path/to/vocoder_config.json
  ```

#### Multi-speaker models

- List the available speakers and choose a `<speaker_id>` among them:

  ```sh
  tts --model_name "<language>/<dataset>/<model_name>"  --list_speaker_idxs
  ```

- Run the multi-speaker TTS model with the target speaker ID:

  ```sh
  tts --text "Text for TTS." --out_path output/path/speech.wav \
      --model_name "<language>/<dataset>/<model_name>"  --speaker_idx <speaker_id>
  ```

- Run your own multi-speaker TTS model:

  ```sh
  tts --text "Text for TTS" --out_path output/path/speech.wav \
      --model_path path/to/model.pth --config_path path/to/config.json \
      --speakers_file_path path/to/speaker.json --speaker_idx <speaker_id>
  ```

#### Voice conversion models

```sh
tts --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>" \
    --source_wav <path/to/speaker/wav> --target_wav <path/to/reference/wav>
```

<!-- end-tts-readme -->
