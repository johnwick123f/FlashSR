# FlashSR

This is a tiny audio super-resolution model based on [hierspeech++](https://github.com/sh-lee-prml/HierSpeechpp) that upscales 16khz audio into much clearer 48khz audio at speed over 200x realtime to 400x realtime!

FlashSR is released under an apache-2.0 license.

## Usage
Simple 1 line installation

```
pip install git+https://github.com/ysharma3501/FlashSR.git
```

Load model
```python
from FastAudioSR import FASR
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(repo_id="YatharthS/FlashSR", filename="upsampler.pth", local_dir=".")
upsampler = FASR(file_path)
```

Run the model
```python
import librosa
import torch
from IPython.display import Audio

y, sr = librosa.load("path/to/audio.wav", sr=16000) ## resamples to 16khz sampling_rate
lowres_wav = torch.from_numpy(y).unsqueeze(0)

new_wav = upsampler.run(lowres_wav)
Audio(new_wav, rate=48000)
```

## Final notes
Thanks very much to the authors of hierspeech++. Thanks for checking out this repository as well.

Stars would be well appreciated, thank you.

Email: yatharthsharma3501@gmail.com
