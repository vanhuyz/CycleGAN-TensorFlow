# CycleGAN-TensorFlow
An implementation of CycleGan using TensorFlow (work in progess).

Original paper: https://arxiv.org/abs/1703.10593

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

Check `$ python dump.py --help` if you want to change default paths.

## Training

```bash
$ python train.py
```

If you want to change some default settings, you can pass those to the command line, such as:

```bash
$ python train.py  \
    --X_train_file=data/tfrecords/horse.tfrecords \
    --Y_train_file=data/tfrecords/zebra.tfrecords
```


Here is list of arguments:
```
usage: train.py [-h] [--batch_size BATCH_SIZE] [--image_size IMAGE_SIZE]
                [--use_lsgan [USE_LSGAN]] [--nouse_lsgan] [--lambda1 LAMBDA1]
                [--lambda2 LAMBDA2] [--learning_rate LEARNING_RATE]
                [--beta1 BETA1] [--X_train_file X_TRAIN_FILE]
                [--Y_train_file Y_TRAIN_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size, default: 1
  --image_size IMAGE_SIZE
                        image size, default: 128
  --use_lsgan [USE_LSGAN]
                        use lsgan (mean squared error) or cross entropy loss,
                        default: True
  --nouse_lsgan
  --lambda1 LAMBDA1     weight for forward cycle loss (X->Y->X), default: 10.0
  --lambda2 LAMBDA2     weight for backward cycle loss (Y->X->Y), default: 10.0
  --learning_rate LEARNING_RATE
                        initial learning rate for Adam, default: 0.0002
  --beta1 BETA1         momentum term of Adam, default: 0.5
  --X_train_file X_TRAIN_FILE
                        X tfrecords file for training,
                        default: data/tfrecords/apple.tfrecords
  --Y_train_file Y_TRAIN_FILE
                        Y tfrecords file for training,
                        default: data/tfrecords/orange.tfrecords
```

Check TensorBoard to see training progress and generated images.

```
$ tensorboard --logdir checkpoints/${datetime}
```

## Samples

Here is some screenshots from TensorBoard.

* apple -> orange

| | |
|-------------------------|-------------------------|
|![apple2orange](samples/apple2orange_1.png) | ![apple2orange](samples/apple2orange_2.png)|


* orange -> apple

| | |
|-------------------------|-------------------------|
|![orang2apple](samples/orange2apple_1.png) | ![orang2apple](samples/orange2apple_2.png)|

## TODO:

* Instance normalization
* Update discriminators using a history of generated images
