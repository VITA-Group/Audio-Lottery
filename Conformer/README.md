# Audio Lottery on Conformer

Our implementation is mainly based on [Efficient Conformer](https://github.com/burchim/EfficientConformer). 
Please find more details about the dependency and usage of the code at the original repo.

## Installation

Install [PyTorch 1.10.1](https://pytorch.org/get-started/locally/)

```
pip install -r requirements.txt
```

Install [ctcdecode](https://github.com/parlance/ctcdecode)

## Download LibriSpeech

[Librispeech](https://www.openslr.org/12) is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.

```
cd datasets
./download_LibriSpeech.sh
```

## Training

You can run an experiment by providing a config file using the '--config_file' flag. Note that **'--prepare_dataset'** and **'--create_tokenizer'** flags may be needed for your **first** experiment.

```
python main_lth.py --config_file configs/EfficientConformerCTCLargeLTH.json
```

The `prune_times` and `prune_percentage` parameters in the config file specifies the number of pruning iterations and the pruning percentage of weights.

Training checkpoints and logs will be saved in the callback folder specified in the config file.
For each pruning iteration, there is a folder, e.g., "prune_0" means training the dense model.
## Evaluation

Models can be evaluated by selecting a subset validation/test mode and by providing the epoch/name of the checkpoint to load for evaluation with the '--initial_epoch' flag. 

With n-gram LM:

```
python main_lth.py --config_file configs/EfficientConformerCTCLargeLTH.json --start_pt pruning_iteration_index --initial_epoch epoch --mode test-clean/test-other
```

Please specific the LM path at the config file `ngram_path`.


## Monitor training

```
tensorboard --logdir callback_path
```

<img src="media/logs.jpg"/>

## LibriSpeech Performance

| Model        			| Remaining weight (%) | test-clean/test-other n-gram WER (%) |
| :-------------------:	|:--------------------:|:------------------------------------:|
| [Dense](https://drive.google.com/drive/folders/1MbkSMyCRKLefFJ9h9SMHgEXGssp0mBtc?usp=sharing)	| 100.0%		| 2.5 / 6.8 |
| [Extreme](https://drive.google.com/drive/folders/1kVHXzkblHbN00dDLBQ16qiKo-2ETZfrh?usp=sharing)	| 16.2%	| 2.4 / 6.4 |
| [Best](https://drive.google.com/drive/folders/1p_a8Fl-omwO7Kturu93dNFr0JtAfTv6K?usp=sharing)	| 51.8% 	| 2.1 / 6.1 |

tokenizer: [here](https://drive.google.com/file/d/1KUiIG5DLp8kx9v4cpYaD_GcKMUUag6xB/view?usp=sharing)

6-gram BPE LM: [here](https://drive.google.com/file/d/12HD87eadwaXnpXlUcylC03ee1ANDu1o6/view?usp=sharing)

Please re-generate the BPE ground-truth with the provided tokenizer by setting `--prepare_dataset` while running 
evaluations with the provided models.


## Reference

```
@inproceedings{ding2021audio,
  title={Audio lottery: Speech recognition made ultra-lightweight, noise-robust, and transferable},
  author={Ding, Shaojin and Chen, Tianlong and Wang, Zhangyang},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
@article{burchi2021efficient,
  title={Efficient conformer: Progressive downsampling and grouped attention for automatic speech recognition},
  author={Burchi, Maxime and Vielzeuf, Valentin},
  journal={arXiv preprint arXiv:2109.01163},
  year={2021}
}
```
