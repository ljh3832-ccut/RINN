# Training Re-parameterizable Integral neural networks.

## Imagenet
```bash
python imagenet.py --integral <IMAGENET FOLDER PATH>
```
Add --data-parallel to use DataParallel training.

# Evaluation of trained INNs.
To resample (prune) and evaluate the integral model run commands below:

## Imagenet
```bash
python imagenet.py --integral --resample --evaluate --checkpoint <INTEGRAL MODEL CHECKPOINT> <IMAGENET FOLDER PATH> 
```


### [Checkpoints of trained integral models are available at link][checkpoints_link].


[checkpoints_link]: https://drive.google.com/drive/folders/1te2HQyCNEIRmN1RbPL2alVN4N-6gT8P9?usp=sharing
