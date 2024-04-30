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
import cv2
import torchmetrics
import torch

## Load validation set
def openValSet(directory):
    with open(directory + 'val.pkl', 'rb') as f:
            val = pickle.load(f)
    X_val, Y_val, X_val_mask, Y_val_mask = val['X_val'], val['Y_val'], val['X_val_mask'], val['Y_val_mask']
    print('X_val shape: ' + str(np.shape(X_val)))
    print('Y_val shape: ' + str(np.shape(Y_val)))
    print('X_val_mask shape: ' + str(np.shape(X_val_mask)))
    print('Y_val_mask shape: ' + str(np.shape(Y_val_mask)))
    
    return X_val, Y_val, X_val_mask, Y_val_mask

## Get Slot Flags
def getSetFlags(batch_size = 512):    
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
 
def getObjectFlags(batch_size = 32):    
    model_type = 'object'

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
def buildLabelDataset(version_str="11.0.0", preds_location='/scratch/pdt9929/DL_Final_Project/dataset/val_predictions/'):
    class LabelDataset(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version(version_str)
        RELEASE_NOTES = {version_str : "Val preds release."}
        SUPPORTED_VERSIONS = [tfds.core.Version(version_str)]

        def _get_num_samples(self):
            # Only last predicted frames
            self.num_samples = 1 * len(self.video_paths)

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
                # centered mask index
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
            print(mask_object_list)
            return mask_object_list

        def _generate_examples(self):
            batch_data = []
            for path in self.video_paths:
                ## Only last predicted frames (for now)
                for i in range(1):
                    image_path = os.path.join(path, f'image_{i+21}.png')
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
## Change val_predictions to unlabeled if you want the unlabeled set !!
## And val_slot_preds to unlabeled set !!
def getSetPreds(FLAGS,
                 val_folder = '/scratch/pdt9929/DL_Final_Project/dataset/val_predictions/',
                 slot_checkpoint_path = '/scratch/pdt9929/google-research/slot_attention/weights.ckpt',
                 slot_save_dir = '/scratch/pdt9929/DL_Final_Project/google_research/slot_attention/label_dataset/val_slot_preds.npy',
                 ):
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
    
    # Get slot predictions
    slot_data_iterator = data_utils.build_clevr_iterator(
        # test split == unlabeled data in our repo
        batch_size, split="prediction", resolution=resolution, shuffle=False,
        max_n_objects=10, get_properties=False, apply_crop=False, unlabeled=True)

    slot_model = model_utils.build_model(resolution, batch_size, num_slots,
                                    num_iterations, model_type="set_prediction")
    
    #slot_model.load_weights(slot_checkpoint_path)
    
    # Dummy optmizer so checkpoint manager works
    optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)
    
    global_step = tf.Variable(
      0, trainable=False, name="global_step", dtype=tf.int64)
    ckpt = tf.train.Checkpoint(
        network=slot_model, optimizer=optimizer, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=ckpt, directory=FLAGS.model_dir, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from %s", ckpt_manager.latest_checkpoint)
    else:
        print("Initializing from scratch.")
    
    slot_predictions = {}
    
    slot_batches = int(np.ceil(len(os.listdir(val_folder))/batch_size))
    
    for _ in tqdm(range(slot_batches), leave=False):
        batch = next(slot_data_iterator)
        slot_preds = slot_model(batch["image"], training=True)
        for idx, filename in enumerate(batch['file_name']):
            slot_predictions[tf.get_static_value(filename)] = [slot_preds[idx]]
            
    with open(slot_save_dir, 'wb') as f:
        np.save(f, np.array(slot_predictions))
        
    return slot_predictions

def getObjectPreds(FLAGS,
                 val_folder = '/scratch/pdt9929/DL_Final_Project/dataset/val_predictions/',
                 object_checkpoint_path = '/scratch/pdt9929/google-research/slot_attention/weights.ckpt',
                 object_save_dir = '/scratch/pdt9929/DL_Final_Project/google_research/slot_attention/label_dataset/val_object_preds.npy',
                 ):
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
    
    # Get Object predictions       
    object_data_iterator = data_utils.build_clevr_iterator(
        # test split == unlabeled data in our repo
        batch_size, split="prediction", resolution=resolution, shuffle=False,
        max_n_objects=10, get_properties=False, apply_crop=False, unlabeled=True)

    object_model = model_utils.build_model(resolution, batch_size, num_slots,
                                    num_iterations, model_type="object_discovery")

    #object_model.load_weights(object_checkpoint_path)
    
    # Dummy optmizer so checkpoint manager works
    optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)
    
    global_step = tf.Variable(
      0, trainable=False, name="global_step", dtype=tf.int64)
    ckpt = tf.train.Checkpoint(
        network=object_model, optimizer=optimizer, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=ckpt, directory=FLAGS.model_dir, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from %s", ckpt_manager.latest_checkpoint)
    else:
        print("Initializing from scratch.")

    object_predictions = {}

    object_batches = int(np.ceil(len(os.listdir(val_folder))/batch_size))

    for _ in tqdm(range(object_batches), leave=False):
        batch = next(object_data_iterator)
        object_preds = object_model(batch["image"], training=True)
        recon_combined, recons, masks, slots = object_preds
        for idx, filename in enumerate(batch['file_name']):
            object_predictions[tf.get_static_value(filename)] = {
                'recon_combined': recon_combined[idx],
                'recon': recons[idx],
                'mask': masks[idx],
                'slots': slots[idx]
            }

    with open(object_save_dir, 'wb') as f:
        np.save(f, np.array(object_predictions))
        
    return object_predictions

def process_targets(target):
    """Unpacks the target into the CLEVR properties."""
    coords = target[:3]
    object_size = tf.argmax(target[3:5])
    material = tf.argmax(target[5:7])
    shape = tf.argmax(target[7:10])
    color = tf.argmax(target[10:18])
    real_obj = target[18]
    return coords, object_size, material, shape, color, real_obj

def getClassDict():
    code_characteristics = { 1:{'shape':'cube', 'material':'metal', 'color': 'gray', 'size': 'small'},
                         2:{'shape':'cube', 'material':'metal', 'color': 'red', 'size': 'small'},
                         3:{'shape':'cube', 'material':'metal', 'color': 'blue', 'size': 'small'},
                         4:{'shape':'cube', 'material':'metal', 'color': 'green', 'size': 'small'}, 
                         5:{'shape':'cube', 'material':'metal', 'color': 'brown', 'size': 'small'}, 
                         6:{'shape':'cube', 'material':'metal', 'color': 'cyan', 'size': 'small'},
                         7:{'shape':'cube', 'material':'metal', 'color': 'purple', 'size': 'small'},
                         8:{'shape':'cube', 'material':'metal', 'color': 'yellow', 'size': 'small'},
                         9:{'shape':'cube', 'material':'rubber', 'color': 'gray', 'size': 'small'},
                        10:{'shape':'cube', 'material':'rubber', 'color': 'red', 'size': 'small'},
                        11:{'shape':'cube', 'material':'rubber', 'color': 'blue', 'size': 'small'},
                        12:{'shape':'cube', 'material':'rubber', 'color': 'green', 'size': 'small'}, 
                        13:{'shape':'cube', 'material':'rubber', 'color': 'brown', 'size': 'small'}, 
                        14:{'shape':'cube', 'material':'rubber', 'color': 'cyan', 'size': 'small'},
                        15:{'shape':'cube', 'material':'rubber', 'color': 'purple', 'size': 'small'},
                        16:{'shape':'cube', 'material':'rubber', 'color': 'yellow', 'size': 'small'}, 
                        17:{'shape':'sphere', 'material':'metal', 'color': 'gray', 'size': 'small'},
                        18:{'shape':'sphere', 'material':'metal', 'color': 'red', 'size': 'small'},
                        19:{'shape':'sphere', 'material':'metal', 'color': 'blue', 'size': 'small'},
                        20:{'shape':'sphere', 'material':'metal', 'color': 'green', 'size': 'small'}, 
                        21:{'shape':'sphere', 'material':'metal', 'color': 'brown', 'size': 'small'}, 
                        22:{'shape':'sphere', 'material':'metal', 'color': 'cyan', 'size': 'small'},
                        23:{'shape':'sphere', 'material':'metal', 'color': 'purple', 'size': 'small'},
                        24:{'shape':'sphere', 'material':'metal', 'color': 'yellow', 'size': 'small'},
                        25:{'shape':'sphere', 'material':'rubber', 'color': 'gray', 'size': 'small'},
                        26:{'shape':'sphere', 'material':'rubber', 'color': 'red', 'size': 'small'},
                        27:{'shape':'sphere', 'material':'rubber', 'color': 'blue', 'size': 'small'},
                        28:{'shape':'sphere', 'material':'rubber', 'color': 'green', 'size': 'small'}, 
                        29:{'shape':'sphere', 'material':'rubber', 'color': 'brown', 'size': 'small'}, 
                        30:{'shape':'sphere', 'material':'rubber', 'color': 'cyan', 'size': 'small'},
                        31:{'shape':'sphere', 'material':'rubber', 'color': 'purple', 'size': 'small'},
                        32:{'shape':'sphere', 'material':'rubber', 'color': 'yellow', 'size': 'small'}, 
                        33:{'shape':'cylinder', 'material':'metal', 'color': 'gray', 'size': 'small'},
                        34:{'shape':'cylinder', 'material':'metal', 'color': 'red', 'size': 'small'},
                        35:{'shape':'cylinder', 'material':'metal', 'color': 'blue', 'size': 'small'},
                        36:{'shape':'cylinder', 'material':'metal', 'color': 'green', 'size': 'small'}, 
                        37:{'shape':'cylinder', 'material':'metal', 'color': 'brown', 'size': 'small'}, 
                        38:{'shape':'cylinder', 'material':'metal', 'color': 'cyan', 'size': 'small'},
                        39:{'shape':'cylinder', 'material':'metal', 'color': 'purple', 'size': 'small'},
                        40:{'shape':'cylinder', 'material':'metal', 'color': 'yellow', 'size': 'small'},
                        41:{'shape':'cylinder', 'material':'rubber', 'color': 'gray', 'size': 'small'},
                        42:{'shape':'cylinder', 'material':'rubber', 'color': 'red', 'size': 'small'},
                        43:{'shape':'cylinder', 'material':'rubber', 'color': 'blue', 'size': 'small'},
                        44:{'shape':'cylinder', 'material':'rubber', 'color': 'green', 'size': 'small'}, 
                        45:{'shape':'cylinder', 'material':'rubber', 'color': 'brown', 'size': 'small'}, 
                        46:{'shape':'cylinder', 'material':'rubber', 'color': 'cyan', 'size': 'small'},
                        47:{'shape':'cylinder', 'material':'rubber', 'color': 'purple', 'size': 'small'},
                        48:{'shape':'cylinder', 'material':'rubber', 'color': 'yellow', 'size': 'small'},
                        }
    return code_characteristics

def getLabelAndColor(class_dict, object_size, material, shape, color):
    shapes = ['cube', 'sphere', 'cylinder']
    materials = ['metal', 'rubber']
    colors = ['gray', 'red', 'blue', 'green', 'brown', 'cyan', 'purple', 'yellow']
    sizes = ['small']
    
    value = {
        'shape': shapes[shape], 
        'material': materials[material], 
        'color': colors[color], 
        'size': sizes[0]
    }
    
    key_list = list(class_dict.keys())
    value_list = list(class_dict.values())
    
    return key_list[value_list.index(value)], colors[color]

def get_camera_params():
    camera_params = {
        'fov': 49.9,  # Field of view in degrees
        'camera_position': [3, 3, 6],  # Camera position (x, y, z)
        'camera_rotation': [-25, 25, 0],  # Camera rotation in degrees (pitch, roll, yaw)
        'camera_sensor_width': 36,  # Camera sensor width in mm
        'camera_sensor_height': 24,  # Camera sensor height in mm
        'image_resolution': [320, 240]  # Image resolution (width, height)
    }
    
    # Calculate the camera rotation matrix
    pitch = camera_params['camera_rotation'][0] * np.pi / 180
    roll = camera_params['camera_rotation'][1] * np.pi / 180
    yaw = camera_params['camera_rotation'][2] * np.pi / 180

    rotation_matrix = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])
    
    return camera_params, rotation_matrix

def world_to_pixel(world_coords, camera_params, rotation_matrix):
    # Extract camera parameters
    camera_position = np.array(camera_params['camera_position'])
    fov = camera_params['fov']
    width, height = camera_params['image_resolution']
    pixel_width = camera_params['camera_sensor_width'] / width
    pixel_height = camera_params['camera_sensor_height'] / height

    # Calculate the direction vector from the camera position to the 3D point
    direction = (world_coords - camera_position).astype(np.float64)

    # Normalize the direction vector
    direction /= np.linalg.norm(direction)

    # Apply the inverse rotation matrix to align with the camera's view
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)
    direction = np.dot(inv_rotation_matrix, direction)

    # Project the 3D point onto the 2D image plane
    # Assuming the camera is looking along the z-axis, ignore the z-component
    u = direction[0] * fov / 2
    v = direction[1] * fov / 2

    # Convert from camera coordinates to pixel indices
    x_index = int((u / pixel_width) + width / 2)
    y_index = int((v / pixel_height) + height / 2)

    return x_index, y_index

def world_to_pixel2(world_coords, camera_params, rotation_matrix):
    camera_position = np.array(camera_params['camera_position'])
    fov = camera_params['fov']
    width, height = camera_params['image_resolution']
    pixel_width = camera_params['camera_sensor_width'] / width
    pixel_height = camera_params['camera_sensor_height'] / height

    # Vector from camera to point in world coordinates
    direction = (world_coords - camera_position).astype(np.float64)

    # Apply the inverse rotation to get the direction in camera coordinates
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)
    camera_coords = np.dot(inv_rotation_matrix, direction)

    # Perspective projection
    # Assuming the camera's z-axis is the viewing direction
    z = camera_coords[2]
    u = (camera_coords[0] / z) * (fov / 2)  # Projecting x
    v = (camera_coords[1] / z) * (fov / 2)  # Projecting y

    # Convert from camera coordinates to pixel indices
    x_index = int((u / pixel_width) + width / 2)
    y_index = int((v / pixel_height) + height / 2)

    return x_index, y_index

## Transform predicted masks back to normal configuration
def generateMasks(set_predictions, object_predictions, class_dict):
    camera_params, rotation_matrix = get_camera_params()
    
    final_masks = {}
    for filename in tqdm(set_predictions, leave=False):
        image = cv2.imread(filename.decode('UTF-8'))
        # 10x19 -- slots x characteristics
        sets = set_predictions[filename][0]
        objects = object_predictions[filename]
        ## recon_combined - 160x240x3; recon - 10x160x240x3; mask - 10x160x240x1; slots - 10x64
        recon_combined, recon, masks, slots = objects['recon_combined'], objects['recon'], objects['mask'], objects['slots']
        
        test_image = {'image': image, 'sets': sets, 'objects': objects}
        
        with open('/scratch/pdt9929/test_image.npy', 'wb') as f:
            np.save(f, test_image)
        
        binary_masks = []
        # Get binary masks from alpha mask
        for mask in masks:
            threshold, bin_mask = cv2.threshold(tf.get_static_value(mask), 0.25, 1, cv2.THRESH_BINARY)
            binary_masks.append(bin_mask)

        summed_mask = np.zeros((160,240))
        for slot in sets:
            coords, object_size, material, shape, color, real_obj = process_targets(slot)
            
            coords = tf.get_static_value(coords)*6-3 #convert to [-3,3] range
            object_size = tf.get_static_value(object_size)
            material = tf.get_static_value(material)
            shape = tf.get_static_value(shape)
            color = tf.get_static_value(color)
            real_obj = tf.get_static_value(real_obj)
            
            pixel_center = world_to_pixel(coords, camera_params, rotation_matrix)
            # threshold for matching
            real_obj_thresh = .75
            
            # switch table for matching slots to mask
            if real_obj > real_obj_thresh:
                correct_label, color_str = getLabelAndColor(class_dict, object_size, material, shape, color)
                for proposal in binary_masks:
                    if proposal[pixel_center] == 1:
                        #print('this is hit')
                        pixels_to_fill = proposal == 1
                        proposal[pixels_to_fill] = correct_label
                        summed_mask = np.where(summed_mask, summed_mask, proposal)
        
        final_masks[filename] = summed_mask
    
    return final_masks

## Measure Jacard distance using provided code
def computeJaccard(predictions, ground_truth):
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
    return jaccard(predictions, ground_truth)

## Main
def main(build_label_set=True):
    # Build label dataset
    if build_label_set == True:
        print('>'*35 + ' Building Validation Label Dataset Version=10.0.0 ' + '<'*35)
        buildLabelDataset()
        print('>'*35 + ' Label Dataset Loaded ' + '<'*35)
        
    # Get code dictionary
    class_dict = getClassDict()
    print('>'*35 + ' Class Dict Loaded ' + '<'*35)
    
    # get slot flags, build label dataset & predict masks
    FLAGS = getSetFlags()
    #object_FLAGS = getObjectFlags()
    print('>'*35 + ' Slot Flags Loaded ' + '<'*35)
    
    # Predict slots
    print('>'*35 + ' Predicting Sets for Last Frame ' + '<'*35)
    set_predictions = getSetPreds(FLAGS)
    print('>'*35 + ' Sets Predicted ' + '<'*35)
    
    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)
            
    del_all_flags(FLAGS)
    
    # get slot flags, build label dataset & predict masks
    FLAGS = getObjectFlags()
    #object_FLAGS = getObjectFlags()
    print('>'*35 + ' Object Flags Loaded ' + '<'*35)
        
    # Predict slots
    print('>'*35 + ' Predicting Objects for Last Frame ' + '<'*35)
    object_predictions = getObjectPreds(FLAGS)
    print('>'*35 + ' Objects Predicted ' + '<'*35)
    
    # Generate masks from slots
    print('>'*35 + ' Generating Masks for Last Frame ' + '<'*35)
    mask_predictions = generateMasks(set_predictions, object_predictions, class_dict)
    print('>'*35 + ' Masks Generated ' + '<'*35)
    
    print('>'*35 + ' Loading Val Ground Truth ' + '<'*35)
    X_val, Y_val, X_val_mask, Y_val_mask = openValSet("dataset/")
    ground_truth = torch.tensor(Y_val_mask[:,10].transpose(0, 2, 3, 1)).squeeze(3)
    print('ground truth shape: ' + str(ground_truth.shape))
    print('>'*35 + ' Ground Truth Loaded ' + '<'*35)
    
    print('>'*35 + ' Preparing final predictions ' + '<'*35)
    mask_keys = list(mask_predictions.keys())
    mask_keys.sort()
    final_prediction = torch.tensor(np.zeros((1, 160, 240)))
    for key in tqdm(mask_keys, leave=False):
        final_prediction = torch.cat((final_prediction, torch.tensor(mask_predictions[key]).unsqueeze(0)), axis=0)
        
    final_prediction = final_prediction[1:]
    
    print('>'*35 + ' Calculating Jaccard distance ' + '<'*35)
    accuracy = computeJaccard(final_prediction, ground_truth)
    print(accuracy)

if __name__ == "__main__":
    main(build_label_set=True)