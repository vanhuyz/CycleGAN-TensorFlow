# set hyperparameters here
BATCH_SIZE = 1
IMAGE_SIZE = 256
NGF = 64
X = apple
Y = orange

# for export_graph: default the last directory in checkpoints
CHECKPOINT_DIR = `ls checkpoints | tail -n 1`

# for reference
INPUT_IMG = input_sample.jpg
OUTPUT_IMG = output_sample.jpg
MODEL = $(X)2$(Y).pb

# commands come here
build_data:
	python build_data.py --X_input_dir=data/$(X)2$(Y)/trainA \
                     --Y_input_dir=data/$(X)2$(Y)/trainB \
                     --X_output_file=data/tfrecords/$(X).tfrecords \
                     --Y_output_file=data/tfrecords/$(Y).tfrecords

train:
	python train.py --batch_size=$(BATCH_SIZE) \
                  --image_size=$(IMAGE_SIZE) \
                  --ngf=$(NGF) \
                  --X=data/tfrecords/$(X).tfrecords \
                  --y=data/tfrecords/$(Y).tfrecords

export_graph:
	python export_graph.py --checkpoint_dir=$(CHECKPOINT_DIR) \
                         --XtoY_model=$(X)2$(Y).pb \
                         --YtoX_model=$(Y)2$(X).pb \
                         --image_size=$(IMAGE_SIZE)

inference:
	python inference.py --model=$(MODEL)\
                      --input=$(INPUT_IMG) \
                      --output=$(OUTPUT_IMG) \
                      --image_size=$(IMAGE_SIZE)
