# CycleGAN-TensorFlow
My attempt to implement CycleGan(https://arxiv.org/abs/1703.10593
) using TensorFlow (in progess).

## Environment

* TensorFlow 1.0.0
* Python 3.6.0

## Data preparing

* First, download a dataset, e.g. apple2orange

```bash
$ bash download_dataset.sh apple2orange
```

* Write the dataset to tfrecords

```bash
$ python dump.py
```

## Train

```bash
$ python train.py
```

Check TensorBoard to see training progress and generated images.

```
$ tensorboard --logdir checkpoints/${datetime}
```

## TODO:

* PatchGAN for discriminators
* Instance normalization
* Update discriminators using a history of generated images
* Learning rate decay
* Smooth label?
* Sample images
