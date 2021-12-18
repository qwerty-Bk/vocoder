# Vocoder

### Training

1. Download the repo

``
!git clone https://github.com/qwerty-Bk/vocoder.git
``
2. Install requirements (``!pip install -r requirements.txt`` for colab)

```
%pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
%pip install torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
%pip install wandb
%pip install prettytable
%pip install librosa
```

3.
``
%cd vocoder
``

4. Training loop

``
!python3 train.py
``

### Inference

1. Repeat 1. -- 3. of Training

2. Download the pretrained model

```
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ulGWCTYu-BAa6w2rgmOeGVJAUXsvdC3d' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ulGWCTYu-BAa6w2rgmOeGVJAUXsvdC3d" -O final_model && rm -rf /tmp/cookies.txt
```

4. Prediciton

``
!python3 test.py
``

for mels in ``test_mels`` directory or 

``
!python test_wav.py
``

for wavs in ``test`` directory.