# Adaptively Aligned Image Captioning via Adaptive Attention Time

This repository includes the implementation for [Adaptively Aligned Image Captioning via Adaptive Attention Time](https://arxiv.org/abs/1909.09060).

## Requirements

- Python 3.6
- Java 1.8.0
- PyTorch 1.0
- [cider](https://github.com/ruotianluo/cider)
- [coco-caption](https://github.com/ruotianluo/coco-caption)
- tensorboardX


## Training AAT

### Prepare data (with python2)

See details in `data/README.md`.

(**notes:** Set `word_count_threshold` in `scripts/prepro_labels.py` to 4 to generate a vocabulary of size 10,369.)

You should also preprocess the dataset and get the cache for calculating cider score for [SCST](https://arxiv.org/abs/1612.00563):

```bash
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```
### Training

```bash
$ sh train-aat.sh
```

See `opts.py` for the options.


### Evaluation

```bash
$ CUDA_VISIBLE_DEVICES=0 python eval.py --model log/log_aat_rl/model.pth --infos_path log/log_aat_rl/infos_aat.pkl  --dump_images 0 --dump_json 1 --num_images -1 --language_eval 1 --beam_size 2 --batch_size 100 --split test
```

## Reference

If you find this repo helpful, please consider citing:

```
@inproceedings{huang2019adaptively,
  title = {Adaptively Aligned Image Captioning via Adaptive Attention Time},
  author = {Huang, Lun and Wang, Wenmin and Xia, Yaxian and Chen, Jie},
  booktitle = {Advances in Neural Information Processing Systems 32},
  year={2019}
}
```

## Acknowledgements

This repository is based on [Ruotian Luo](https://github.com/ruotianluo)'s [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).
