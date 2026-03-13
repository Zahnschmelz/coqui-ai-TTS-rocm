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
# download a model (for example xtts-v2.0.3)
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
cd ../..
```
# download a voice .wav (for example de_sample.wav)
```
mkdir -p voices && cd voices
wget https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.3/samples/de_sample.wav?download=true
cd ..
```

## Synthesizing speech by 🐸TTS
<!-- start inference -->
### 🐍 Python API (run python inside coqui-ai-TTS-rocm directory)

#### Multi-speaker and multi-lingual model

```python
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

config = XttsConfig()
config.load_json("./models/xttsv2_2.0.3/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="models/xttsv2_2.0.3/")
model.cuda()
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["voices/de_sample.wav"])

out = model.inference(
    "Hello world!",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    enable_text_splitting=True)

torchaudio.save("outputs/out.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
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
tts = TTS("models/xtts-v2.0.3").to("cuda")
tts.voice_conversion_to_file(
  source_wav="voices/de_sample.wav",
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
