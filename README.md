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
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm7.1 # (check for newest version: https://pytorch.org/get-started/locally/)
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
import torchaudio
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


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
<!-- end-tts-readme -->
