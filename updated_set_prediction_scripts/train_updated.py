# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop for set prediction based on Slot Attention."""
import datetime
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import slot_attention.our_data as data_utils
import slot_attention.model as model_utils
import slot_attention.utils as utils

import tensorflow_datasets as tfds
import numpy as np
import os


FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "/scratch/kmk9461/google-research/slot_attention/set_prediction/",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 512, "Batch size for the model.")
flags.DEFINE_integer("num_slots", 10, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
flags.DEFINE_integer("num_train_steps", 1000, '')#000, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 1000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 50,#0000,
                     "Number of steps for the learning rate decay.")

# Label Data generate
class LabelDataset(tf.data.Dataset):
    def __init__(self, path, batch_size):
        self.video_paths = [os.path.join(path, dir_path) for dir_path in os.listdir(path)]
        self.num_samples = 0

        # Load the mask shape from the first video path
        mask_sample_path = os.path.join(self.video_paths[0], 'mask.npy')
        mask_shapes = np.load(mask_sample_path).shape

        # Calculate the total number of samples
        self.num_samples = mask_shapes[0] * len(self.video_paths)
        self.batch_size = batch_size

    def _inputs(self):
        return []

    def element_spec(self):
        image_shape = tf.TensorShape([None, None, 3])
        mask_shape = tf.TensorShape([None])
        return (image_shape, mask_shape)

    def _generator(self):
        images_batch = []
        masks_batch = []
        for path in self.video_paths:
            mask_path = os.path.join(path, 'mask.npy')
            masks = np.load(mask_path)

            for i in range(masks.shape[0]): 
                image_path = os.path.join(path, f'image_{i}.png')
                image_data = tf.io.read_file(image_path)

                image = tf.image.decode_png(image_data, channels=3)
                mask = tf.convert_to_tensor(masks[i])

                images_batch.append(image)
                masks_batch.append(mask)

                if ((len(images_batch) >= self.batch_size) & (len(masks_batch) >= self.batch_size)):
                    output_images_batch = images_batch[:]
                    output_masks_batch = masks_batch[:]

                    images_batch = []
                    masks_batch = []

                    yield {'image': tf.stack(output_images_batch), 'target': tf.stack(output_masks_batch)}

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        return self._generator()


# NOTE: For simplicity, we do not use `tf.function` compilation in this
# example. In TF graph mode, scipy-based Hungarian matching has to be called
# via `tf.py_function`.
def train_step(batch, model, optimizer):
  """Perform a single training step."""

  # Get the prediction of the models and compute the loss.
  with tf.GradientTape() as tape:
    preds = model(batch["image"], training=True)
    loss_value = utils.hungarian_huber_loss(preds, batch["target"])

  # Get and apply gradients.
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))

  return loss_value


def main(argv):
  del argv
  # Hyperparameters of the model.
  batch_size = FLAGS.batch_size
  num_slots = FLAGS.num_slots
  num_iterations = FLAGS.num_iterations
  base_learning_rate = FLAGS.learning_rate
  num_train_steps = FLAGS.num_train_steps
  warmup_steps = FLAGS.warmup_steps
  decay_rate = FLAGS.decay_rate
  decay_steps = FLAGS.decay_steps
  tf.random.set_seed(FLAGS.seed)
  resolution = (160, 240)

  # Build dataset iterators, optimizers and model.
  data_iterator = data_utils.build_clevr_iterator(
      batch_size, split="train", resolution=resolution, shuffle=True,
      max_n_objects=10, get_properties=True, apply_crop=False)
  data_iterator_validation = data_utils.build_clevr_iterator(
      batch_size, split="train_eval", resolution=resolution, shuffle=False,
      max_n_objects=10, get_properties=True, apply_crop=False)
    
  #data_iterator = LabelDataset('/scratch/kmk9461/DL_Final_Project/dataset/train', batch_size)
  #data_iterator_validation = LabelDataset('/scratch/kmk9461/DL_Final_Project/dataset/train', batch_size)

  optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)

  model = model_utils.build_model(resolution, batch_size, num_slots,
                                  num_iterations, model_type="set_prediction")

  # Prepare checkpoint manager.
  global_step = tf.Variable(
      0, trainable=False, name="global_step", dtype=tf.int64)
  ckpt = tf.train.Checkpoint(
      network=model, optimizer=optimizer, global_step=global_step)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt, directory=FLAGS.model_dir, max_to_keep=5)
  ckpt.restore(ckpt_manager.latest_checkpoint)
  if ckpt_manager.latest_checkpoint:
    logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
  else:
    logging.info("Initializing from scratch.")

  start = time.time()
  for _ in range(num_train_steps):
    batch = next(data_iterator)
    print(_)
    # Learning rate warm-up.
    if global_step < warmup_steps:
      learning_rate = base_learning_rate * tf.cast(
          global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
    else:
      learning_rate = base_learning_rate
    learning_rate = learning_rate * (decay_rate ** (
        tf.cast(global_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
    optimizer.lr = learning_rate.numpy()

    loss_value = train_step(batch, model, optimizer)

    # Update the global step. We update it before logging the loss and saving
    # the model so that the last checkpoint is saved at the last iteration.
    global_step.assign_add(1)

    # Log the training loss and validation average precision.
    # We save the checkpoints every 1000 iterations.
    if not global_step % 100:
      logging.info("Step: %s, Loss: %.6f, Time: %s",
                   global_step.numpy(), loss_value,
                   datetime.timedelta(seconds=time.time() - start))
    if _==999:#not global_step  % 1000:
      # For evaluating the AP score, we get a batch from the validation dataset.
      batch = next(data_iterator_validation)
      preds = model(batch["image"], training=False)
      ap = [
          utils.average_precision_clevr(preds, batch["target"], d)
          for d in [-1., 1., 0.5, 0.25, 0.125]
      ]
      logging.info(
          "AP@inf: %.2f, AP@1: %.2f, AP@0.5: %.2f, AP@0.25: %.2f, AP@0.125: %.2f",
          ap[0], ap[1], ap[2], ap[3], ap[4])

      # Save the checkpoint of the model.
      saved_ckpt = ckpt_manager.save()
      logging.info("Saved checkpoint: %s", saved_ckpt)
        
  model.save_weights('/scratch/kmk9461/google-research/slot_attention/weights.ckpt')


if __name__ == "__main__":
  app.run(main)
