import numpy as np
from tqdm import tqdm
import pickle
import os
import datetime
import time
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from google_research.slot_attention.updated_set_prediction_scripts import our_data as data_utils
from google_research.slot_attention.updated_set_prediction_scripts import model as model_utils
from google_research.slot_attention import utils
from PIL import Image
from math import tan, pi, cos, sin

## Load validation set
def openValSet(directory):
    with open(directory + 'val.pkl', 'rb') as f:
            val = pickle.load(f)
    X_val, Y_val, X_val_mask, Y_val_mask = val['X_val'], val['Y_val'], val['X_val_mask'], val['Y_val_mask']
    print('X_val shape: ' + str(np.shape(X_val)))
    print('Y_val shape: ' + str(np.shape(Y_val)))
    print('X_val_mask shape: ' + str(np.shape(X_val_mask)))
    print('Y_test_mask shape: ' + str(np.shape(X_val_mask)))
    
    return X_val, Y_val, X_val_mask, Y_val_mask

## Get Slot Flags
def getSlotFlags(batch_size = 512):    
    model_type = 'set'

    FLAGS = flags.FLAGS
    ## UNCOMMENT FOR CORRECT FILE LOCATION
    if model_type == 'object':
        flags.DEFINE_string("model_dir", "/scratch/pdt9929/DL_Final_Project/google_research/slot_attention/object_discovery/",
                            "Where to save the checkpoints.")
    elif model_type == 'set':
        flags.DEFINE_string("model_dir", "/scratch/pdt9929/DL_Final_Project/google_research/slot_attention/set_prediction/",
                            "Where to save the checkpoints.")
    flags.DEFINE_integer("seed", 0, "Random seed.")
    flags.DEFINE_integer("batch_size", batch_size, "Batch size for the model.")
    flags.DEFINE_integer("num_slots", 10, "Number of slots in Slot Attention.")
    flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
    flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
    flags.DEFINE_integer("num_train_steps", 10000, '')#000, "Number of training steps.")
    flags.DEFINE_integer("warmup_steps", 1000,
                        "Number of warmup steps for the learning rate.")
    flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
    flags.DEFINE_integer("decay_steps", 500,#0000,
                        "Number of steps for the learning rate decay.")
    
    return FLAGS
    
## Generate Slot Data Iterator
def buildLabelDataset(version_str="10.0.0", preds_location='/scratch/pdt9929/DL_Final_Project/dataset/val_predictions/'):
    class LabelDataset(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version(version_str)
        RELEASE_NOTES = {version_str : "Val preds release."}
        SUPPORTED_VERSIONS = [tfds.core.Version(version_str)]

        def _get_num_samples(self):
            # 11 Predicted frames
            self.num_samples = 11 * len(self.video_paths)

        def _init_camera_params(self):
            # Default camera parameters for the CLEVR dataset
            self.default_camera_params = {
                'fov': 49.9,  # Field of view in degrees
                'camera_position': [3, 3, 6],  # Camera position (x, y, z)
                'camera_rotation': [-25, 25, 0],  # Camera rotation in degrees (pitch, roll, yaw)
                'camera_sensor_width': 36,  # Camera sensor width in mm
                'camera_sensor_height': 24,  # Camera sensor height in mm
                'image_resolution': [320, 240]  # Image resolution (width, height)
            }

            # Calculate the pixel dimensions
            self.width, self.height = self.default_camera_params['image_resolution']
            self.pixel_width = self.default_camera_params['camera_sensor_width'] / self.width
            self.pixel_height = self.default_camera_params['camera_sensor_height'] / self.height

            # Calculate the camera rotation matrix
            pitch = self.default_camera_params['camera_rotation'][0] * pi / 180
            roll = self.default_camera_params['camera_rotation'][1] * pi / 180
            yaw = self.default_camera_params['camera_rotation'][2] * pi / 180

            self.rotation_matrix = np.array([
                [cos(yaw) * cos(pitch), cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll), cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll)],
                [sin(yaw) * cos(pitch), sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll), sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll)],
                [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll)]]
            )

        def _init_mappings(self):

            self.code_characteristics = {}
            counter = 1
            shapes = ['cube', 'sphere', 'cylinder']
            materials = ['metal', 'rubber']
            colors = ['gray', 'red', 'blue', 'green', 'brown', 'cyan', 'purple', 'yellow']
            sizes = ['small']

            for shape in shapes:
                for material in materials:
                    for color in colors:
                        for size in sizes:
                            self.code_characteristics[counter] = {'shape': shape, 'material': material, 'color': color, 'size': size}
                            counter += 1

        def maskpixel_to_3D_coordinates(self, x_index, y_index):
            # Convert 2D pixel coordinates to 3D world coordinates
            u = (x_index - self.width / 2) * self.pixel_width
            v = (y_index - self.height / 2) * self.pixel_height
            direction = np.array([u, v, self.default_camera_params['fov'] / 2])
            direction /= np.linalg.norm(direction)
            direction = np.dot(self.rotation_matrix, direction)
            world_coords = self.default_camera_params['camera_position'] + direction

            return world_coords

        def _inputs(self):
            return []

        def element_spec(self):
            image_shape = tf.TensorShape([None, None, 3])
            mask_shape = tf.TensorShape([None])
            return (image_shape, mask_shape)

        def _process_image(self, image_data):

            image = tf.image.decode_png(image_data, channels=3)
            image = tf.cast(image, tf.float32)
            image = ((image / 255.0) - 0.5) * 2.0  # Rescale to [-1, 1].
            image = tf.image.resize(image, (160,240), method=tf.image.ResizeMethod.BILINEAR)
            image = tf.clip_by_value(image, -1., 1.)

            return image

        def _process_mask(self, mask):
            mask = tf.convert_to_tensor(mask)
            unique_mask_codes = np.unique(mask[mask != 0])

            mask_shape = []
            mask_material = []
            mask_color = []
            mask_size = []
            mask_3d_coords = []
            mask_mask = []
            mask_code = []

            for unique_mask_code in unique_mask_codes:
                unique_mask = mask == unique_mask_code

                true_indices = np.where(unique_mask)
                x_mask, y_mask = (int(np.mean(true_indices[0])), int(np.mean(true_indices[1])))

                mask_object = {}

                mask_shape.append(self.code_characteristics[unique_mask_code]['shape'])
                mask_material.append(self.code_characteristics[unique_mask_code]['material'])
                mask_color.append(self.code_characteristics[unique_mask_code]['color'])
                mask_size.append(self.code_characteristics[unique_mask_code]['size'])
                mask_3d_coords.append(self.maskpixel_to_3D_coordinates(x_mask, y_mask))
                mask_mask.append(unique_mask)
                mask_code.append(unique_mask_code)

            attrs = [
                "color",
                "material",
                "shape",
                "size",
                "3d_coords",
                # "mask",
                # "code",
            ]
            mask_object = {}
            mask_object['shape'] = mask_shape
            mask_object['material'] = mask_material
            mask_object['color'] = mask_color
            mask_object['size'] = mask_size
            mask_object['3d_coords']= mask_3d_coords
            # mask_object['mask'] = mask_mask
            # mask_object['code'] = mask_code

            mask_object_list = [{attr: mask_object[attr][obj] for attr in attrs} for obj in range(len(mask_material))]

            return mask_object_list

        def _generate_examples(self):
            batch_data = []
            for path in self.video_paths:
                ## 11 predicted frames
                for i in range(11):
                    image_path = os.path.join(path, f'image_{i+11}.png')
                    image_data = tf.io.read_file(image_path)
                    image = self._process_image(image_data)
                    record = {'image': image_path, 'file_name': image_path}
                    fname =  image_path

                    yield fname, record

        def _split_generators(self, dl_manager):
            split= [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TEST,
                    gen_kwargs={},
                )
            ]
            return split

        def _info(self):

            self.video_paths = [os.path.join(preds_location, dir_path) for dir_path in os.listdir(preds_location) if dir_path.startswith('video')]
            self.batch_size = 2
            self._init_camera_params()
            self._init_mappings()
            self._get_num_samples()

            features = {
                "image": tfds.features.Image(),
                "file_name": tfds.features.Text()
            }
            features_dict = tfds.features.FeaturesDict(features)
            dataset_info = tfds.core.DatasetInfo(
                builder=self,
                features=features_dict,
            )
            return dataset_info
        
    # Generate label dataset
    path = '/scratch/pdt9929/DL_Final_Project/google_research/slot_attention/'
    os.environ['TFDS_DATA_DIR'] = path
    os.environ['TFDS_MANUAL_DIR'] = path
    custom_dataset_file = LabelDataset()
    custom_dataset_file.download_and_prepare(download_dir = path)

## Get Slot Model & Predict masks
def getSlotModel(FLAGS, checkpoint_path = '/scratch/pdt9929/google-research/slot_attention/weights.ckpt',
                 save_dir = '/scratch/pdt9929/google_research/slot_attention/label_dataset/val_slot_preds.npy'):
    # Hyperparameters of the model.
    FLAGS(sys.argv)
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
    
    data_iterator = data_utils.build_clevr_iterator(
        # test split == unlabeled data in our repo
        batch_size, split="prediction", resolution=resolution, shuffle=False,
        max_n_objects=10, get_properties=False, apply_crop=False, unlabeled=True)

    model = model_utils.build_model(resolution, batch_size, num_slots,
                                    num_iterations, model_type="set_prediction")
    
    model.load_weights(checkpoint_path)
    
    slot_predictions = {}
    
    for _ in tqdm(data_iterator, leave=False):
        batch = next(data_iterator)
        preds = model(batch["image"], training=True)
        slot_predictions['mask'] = preds 
        slot_predictions['file_name'] = batch['file_name']
        
    with open(save_dir, 'wb') as f:
        np.save(f, np.array(slot_predictions))
        
    print('Shape of Slot Predictions: ' + slot_predictions['mask'].shape)
        
    return slot_predictions

## Transform predicted masks back to normal configuration

## Measure Jacard distance using provided code

## Main
def main(build_label_set=True):
    # get slot flags, build label dataset & predict masks
    FLAGS = getSlotFlags()
    print('>'*35 + ' Slot Flags Loaded ' + '<'*35)
    
    # Build label dataset
    if build_label_set == True:
        print('>'*35 + ' Building Validation Label Dataset Version=10.0.0 ' + '<'*35)
        buildLabelDataset()
        print('>'*35 + ' Label Dataset Loaded ' + '<'*35)
        
    # Predict masks
    print('>'*35 + ' Predicting Masks for Last 11 Frames with Slot Attention ' + '<'*35)
    slot_predictions = getSlotModel(FLAGS)
    print('>'*35 + ' Slots Predicted ' + '<'*35)

if __name__ == "__main__":
    main(build_label_set=True)