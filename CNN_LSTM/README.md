# Audio lottery on CNN+LSTM

Our implementation is based on the [DeepSpeech2 PyTorch implementation](https://github.com/SeanNaren/deepspeech.pytorch), folked at [this](https://github.com/SeanNaren/deepspeech.pytorch/commit/78f7fb791f42c44c8a46f10e79adad796399892b) commit.

## Installation

The instructions are obtained from the original repo.

Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

Install this fork for Warp-CTC bindings:
```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc; mkdir build; cd build; cmake ..; make
export CUDA_HOME="/usr/local/cuda"
cd ../pytorch_binding && python setup.py install
```

Install NVIDIA apex:
```
git clone --recursive https://github.com/NVIDIA/apex.git
cd apex && pip install .
```

If you want decoding to support beam search with an optional language model, install ctcdecode:
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

Finally clone this repo and run this within the repo:
```
pip install -r requirements.txt
pip install -e . # Dev install
```

## Training

### Datasets

Pre-process [LibriSpeech](https://www.openslr.org/12) dataset.


```
cd data/
python librispeech.py
```

### Training a Model

Run

```
python prune_train.py +experiment=librispeech_prune
```

Distributed training not tested.

SpecAug and Noise injection functions are the same as the base repo.



## Testing/Inference

### Inference with 3-gram LM

Edit the config files `experiment/librispeech_prune_eval_clean.yaml` or `experiment/librispeech_prune_eval_other.yaml`.

Then, run

```
python prune_test.py +experiment=librispeech_prune_eval_clean
```

## LibriSpeech Performance

| Model        			| Remaining weight (%) | test-clean/test-other n-gram WER (%) |
| :-------------------:	|:--------------------:|:------------------------------------:|
| [Dense](https://drive.google.com/drive/folders/160X8_KRRwIoz4bFhzyTUeGcHwlJQ_eHM?usp=sharing)	| 100.0%		| 7.9 / 21.0 |
| [Extreme](https://drive.google.com/drive/folders/1vYNgc_VwfNJIsYB5hCNInr5gs_0uxu2x?usp=sharing)	| 21.0%	| 7.9 / 20.5 |
| [Best](https://drive.google.com/drive/folders/1ynX7Zc8jBA4whxcjVKcivJwpdVZBG6HB?usp=sharing)	| 51.8% 	| 7.1 / 19.2 |

3-gram grapheme LM: [here](https://drive.google.com/file/d/1tqPrr1eV1fUinOaaJH5yNqYh6512nyCN/view?usp=sharing)
## Reference

```
@inproceedings{ding2021audio,
  title={Audio lottery: Speech recognition made ultra-lightweight, noise-robust, and transferable},
  author={Ding, Shaojin and Chen, Tianlong and Wang, Zhangyang},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
@inproceedings{amodei2016deep,
  title={Deep speech 2: End-to-end speech recognition in english and mandarin},
  author={Amodei, Dario and Ananthanarayanan, Sundaram and Anubhai, Rishita and Bai, Jingliang and Battenberg, Eric and Case, Carl and Casper, Jared and Catanzaro, Bryan and Cheng, Qiang and Chen, Guoliang and others},
  booktitle={International conference on machine learning},
  pages={173--182},
  year={2016},
  organization={PMLR}
}
```
