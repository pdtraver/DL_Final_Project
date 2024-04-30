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

def buildUnlabeledDataset(version_str="31.0.0", preds_location='/scratch/pdt9929/DL_Final_Project/dataset/unlabeled/'):
    class LabelDataset(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version(version_str)
        RELEASE_NOTES = {version_str : "Unlabeled mask preds release."}
        SUPPORTED_VERSIONS = [tfds.core.Version(version_str)]

        def _get_num_samples(self):
            # Only last predicted frames
            self.num_samples = 22 * len(self.video_paths)

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
                for i in range(22):
                    image_path = os.path.join(path, f'image_{i}.png')
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
def getSlotModel(FLAGS, val_folder = '/scratch/pdt9929/DL_Final_Project/dataset/val_predictions/',
                 checkpoint_path = '/scratch/pdt9929/google-research/slot_attention/weights.ckpt',
                 save_dir = '/scratch/pdt9929/DL_Final_Project/google_research/slot_attention/label_dataset/val_slot_preds.npy'):
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
    
    batches = int(np.ceil(len(os.listdir(val_folder))/batch_size))
    
    for _ in tqdm(range(batches), leave=False):
        batch = next(data_iterator)
        preds = model(batch["image"], training=True)
        for idx, filename in enumerate(batch['file_name']):
            slot_predictions[tf.get_static_value(filename)] = [preds[idx]]
       
    # print('Shape of slot predictions: ')
    # print(slot_predictions['mask'][0].shape)
    # print(slot_predictions['mask'][0][0])
    # print(slot_predictions['file_name'][0][0])
    with open(save_dir, 'wb') as f:
        np.save(f, np.array(slot_predictions))
        
    return slot_predictions

def getUnlabeled(FLAGS, val_folder = '/scratch/pdt9929/DL_Final_Project/dataset/unlabeled/',
                 checkpoint_path = '/scratch/pdt9929/google-research/slot_attention/weights.ckpt',
                 save_dir = '/scratch/pdt9929/DL_Final_Project/google_research/slot_attention/label_dataset/unlabeled_masks.npy'):
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
    
    # data_iterator = data_utils.build_clevr_iterator(
    #     # test split == unlabeled data in our repo
    #     batch_size, split="prediction", resolution=resolution, shuffle=False,
    #     max_n_objects=10, get_properties=False, apply_crop=False, unlabeled=True)

    model = model_utils.build_model(resolution, batch_size, num_slots,
                                    num_iterations, model_type="set_prediction")
    
    model.load_weights(checkpoint_path)
    
    slot_predictions = {}
    
    #batches = int(np.ceil(len(os.listdir(val_folder))/batch_size))
    
    def _process_unlabeled_images(path):
        processed_images = []
        file_names = []
        for i in range(len(os.listdir(os.path.join(val_folder, path)))):
            image_path = os.path.join(val_folder, path, f'image_{i}.png')
            image_data = tf.io.read_file(image_path)

            image = tf.image.decode_png(image_data, channels=3)
            image = tf.cast(image, tf.float32)
            image = ((image / 255.0) - 0.5) * 2.0  # Rescale to [-1, 1].
            image = tf.image.resize(image, (160,240), method=tf.image.ResizeMethod.BILINEAR)
            image = tf.clip_by_value(image, -1., 1.)
            processed_images.append(tf.expand_dims(image, 0))
            file_names.append(image_path)
            
        batch = {'image': tf.concat(processed_images, 0), 'file_name': file_names}
        
        return batch
    
    # batches are each video (22)
    batches = os.listdir(val_folder)
    intermediate_save_loc = '/scratch/pdt9929/DL_Final_Project/dataset/unlabeled_slot_predictions'
    
    for folder in tqdm(batches, leave=False):
        write_path = os.path.join(intermediate_save_loc, folder)
        if not os.path.exists(write_path):
            os.mkdir(write_path)
        else:
            # if directory already exists, it means it's been computed before, so just continue
            continue
        
        batch = _process_unlabeled_images(folder)
        preds = model(batch['image'], training=True)
            
        with open(os.path.join(write_path, 'slot_preds.npy'), 'wb') as f:
            np.save(f, np.array(tf.get_static_value(preds)))
            
        for idx, filename in enumerate(batch['file_name']):
            slot_predictions[filename] = [preds[idx]]
       
    # print('Shape of slot predictions: ')
    # print(slot_predictions['mask'][0].shape)
    # print(slot_predictions['mask'][0][0])
    # print(slot_predictions['file_name'][0][0])
    with open(save_dir, 'wb') as f:
        np.save(f, np.array(slot_predictions))
        
    return slot_predictions

# Helper function for translation from slot to masks
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

# Default camera params
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

class ColorMasker:
    def __init__(self, image):
        self.image = image
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.color_ranges = {
            'gray': ((0, 0, 40), (30, 20, 100)),
            'red': (((0, 100, 100), (10, 255, 255)), ((160, 100, 100), (180, 255, 255))),
            'blue': ((110, 50, 50), (130, 255, 255)),
            'green': ((30, 70, 60), (60, 255, 255)),
            'brown': ((10, 100, 20), (20, 255, 200)),
            'cyan': ((80, 90, 60), (100, 255, 255)),
            'purple': ((130, 50, 50), (160, 255, 255)),
            'yellow': ((25, 100, 100), (30, 255, 255))
        }

    def separate_objects(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                temp_mask = np.zeros_like(mask)
                cv2.drawContours(temp_mask, [contour], -1, 255, thickness=cv2.FILLED)
                masks.append(temp_mask)
        return masks

    def get_mask(self, color_code):
        if color_code not in self.color_ranges:
            raise ValueError("Invalid color code")
        
        ranges = self.color_ranges[color_code]
        if color_code == 'red':  # Red has two ranges
            mask1 = cv2.inRange(self.hsv_image, np.array(ranges[0][0], dtype=np.uint8), np.array(ranges[0][1], dtype=np.uint8))
            mask2 = cv2.inRange(self.hsv_image, np.array(ranges[1][0], dtype=np.uint8), np.array(ranges[1][1], dtype=np.uint8))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(self.hsv_image, np.array(ranges[0], dtype=np.uint8), np.array(ranges[1], dtype=np.uint8))
        
        return self.separate_objects(mask)
        
    def get_mask_arg(self, mask, predicted_centers):
        true_indices = np.where(mask)
        x_mask, y_mask = (int(np.mean(true_indices[0])), int(np.mean(true_indices[1])))
        
        # Convert predicted_centers to a numpy array if it's not already one
        predicted_centers = np.array(predicted_centers)
        
        # Calculate the squared Euclidean distances to all predicted centers
        distances = np.sum((predicted_centers - np.array([x_mask, y_mask]))**2, axis=1)

        # Find the index of the closest center
        closest_index = np.argmin(distances)

        # Get the closest center coordinates
        closest_center = predicted_centers[closest_index]

        return closest_center

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

## Transform predicted masks back to normal configuration
def generateMasks(slot_predictions, class_dict):
    camera_params, rotation_matrix = get_camera_params()
    
    for filename in slot_predictions:
        image = cv2.imread(filename.decode('UTF-8'))
        slots = slot_predictions[filename][0]
        
        #Masking protocol
        color_mask = ColorMasker(image)
        
        color_masks = {}
        for color in color_mask.color_ranges.keys():
            color_masks[color] = color_mask.get_mask(color)

        masks_to_sum = []
        for slot in slots:
            coords, object_size, material, shape, color, real_obj = process_targets(slot)
            
            coords = tf.get_static_value(coords)*6 - 3 #convert to [0,6] range
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
                mask_proposals = color_masks[color_str]
                for proposal in mask_proposals:
                    if proposal[pixel_center] == 1:
                        print('this is hit')
                        pixels_to_fill = proposal == 1
                        proposal[pixels_to_fill] = correct_label
                        masks_to_sum.append(proposal)
            
## Measure Jacard distance using provided code
def computeJaccard(predictions, ground_truth):
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
    return jaccard(predictions, ground_truth)

# Calculate the Jaccard distance between two binary masks.
def jaccard_distance(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    jaccard_index = np.sum(intersection) / np.sum(union)
    jaccard_dist = 1 - jaccard_index
    return jaccard_dist

# Calculate the Jaccard distance between two binary masks or batches of masks.
def jaccard_distance_tensorflow(mask1, mask2):
    intersection = tf.math.logical_and(mask1, mask2)
    union = tf.math.logical_or(mask1, mask2)

    intersection_sum = tf.math.reduce_sum(tf.cast(intersection, tf.float32), axis=[1, 2, 3])
    union_sum = tf.math.reduce_sum(tf.cast(union, tf.float32), axis=[1, 2, 3])

    jaccard_index = tf.math.divide_no_nan(intersection_sum, union_sum)
    jaccard_dist = 1.0 - jaccard_index

    return jaccard_dist

## Main
def main(build_label_set=True):
    # get slot flags, build label dataset & predict masks
    FLAGS = getSlotFlags()
    print('>'*35 + ' Slot Flags Loaded ' + '<'*35)
    
    # # Build label dataset
    # if build_label_set == True:
    #     print('>'*35 + ' Building Validation Label Dataset Version=10.0.0 ' + '<'*35)
    #     buildLabelDataset()
    #     print('>'*35 + ' Label Dataset Loaded ' + '<'*35)
        
    # # Predict slots
    # print('>'*35 + ' Predicting Slots for Last Frame ' + '<'*35)
    # slot_predictions = getSlotModel(FLAGS)
    # print('>'*35 + ' Slots Predicted ' + '<'*35)
    
    # # Get code dictionary
    # class_dict = getClassDict()
    # print('>'*35 + ' Class Dict Loaded ' + '<'*35)
    
    # # Generate masks from slots
    # print('>'*35 + ' Generating Masks for Last Frame ' + '<'*35)
    # mask_predictions = generateMasks(slot_predictions, class_dict)
    # print('>'*35 + ' Masks Generated ' + '<'*35)
    
    # X_val, Y_val, X_val_mask, Y_val_mask = openValSet("dataset/")
    
    #buildUnlabeledDataset()
    slot_predictions = getUnlabeled(FLAGS)

if __name__ == "__main__":
    main(build_label_set=True)