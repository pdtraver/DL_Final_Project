import torch
from tqdm import tqdm
import pickle
import numpy as np

def convert_to_multi_hot(masks):
    # Dimensions: B, F, C, H, W (assuming masks are already loaded as such)
    batch_size, num_frames, channels, height, width = masks.shape
    num_shapes = 3
    num_materials = 2
    num_colors = 8
    # Create an output tensor for the multi-hot encoding
    multi_hot = np.zeros((batch_size, num_frames, (num_shapes + num_materials + num_colors + 1), height, width), dtype=np.uint8)
    
    code_characteristics = getClassDict()
    # Encode each attribute in its respective channel
    for b in tqdm(range(batch_size), leave=False):
        for f in range(num_frames):
            for y in range(height):
                for x in range(width):
                    label = masks[b, f, 0, y, x].item()
                    if label == 0:  # Assuming 0 is background and has no attributes
                        multi_hot[b, f, num_shapes + num_materials + num_colors, y, x] = 1
                        continue
                    attributes = code_characteristics[label]
                    shape_index = {'cube': 0, 'sphere': 1, 'cylinder': 2}[attributes['shape']]
                    material_index = {'metal': 0, 'rubber': 1}[attributes['material']]
                    color_index = {'gray': 0, 'red': 1, 'blue': 2, 'green': 3, 'brown': 4, 'cyan': 5, 'purple': 6, 'yellow': 7}[attributes['color']]
                    
                    # Set the bits for shape, material, and color
                    multi_hot[b, f, shape_index, y, x] = 1
                    multi_hot[b, f, num_shapes + material_index, y, x] = 1
                    multi_hot[b, f, num_shapes + num_materials + color_index, y, x] = 1

    return multi_hot

def getClassDict():
    code_characteristics = {1:{'shape':'cube', 'material':'metal', 'color': 'gray', 'size': 'small'},
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

def getReverseDict():
    code_characteristics = {(('shape', 'cube'), ('material', 'metal'), ('color', 'gray'), ('size', 'small')): 1, 
                            (('shape', 'cube'), ('material', 'metal'), ('color', 'red'), ('size', 'small')): 2, 
                            (('shape', 'cube'), ('material', 'metal'), ('color', 'blue'), ('size', 'small')): 3, 
                            (('shape', 'cube'), ('material', 'metal'), ('color', 'green'), ('size', 'small')): 4, 
                            (('shape', 'cube'), ('material', 'metal'), ('color', 'brown'), ('size', 'small')): 5, 
                            (('shape', 'cube'), ('material', 'metal'), ('color', 'cyan'), ('size', 'small')): 6, 
                            (('shape', 'cube'), ('material', 'metal'), ('color', 'purple'), ('size', 'small')): 7, 
                            (('shape', 'cube'), ('material', 'metal'), ('color', 'yellow'), ('size', 'small')): 8, 
                            (('shape', 'cube'), ('material', 'rubber'), ('color', 'gray'), ('size', 'small')): 9, 
                            (('shape', 'cube'), ('material', 'rubber'), ('color', 'red'), ('size', 'small')): 10, 
                            (('shape', 'cube'), ('material', 'rubber'), ('color', 'blue'), ('size', 'small')): 11, 
                            (('shape', 'cube'), ('material', 'rubber'), ('color', 'green'), ('size', 'small')): 12, 
                            (('shape', 'cube'), ('material', 'rubber'), ('color', 'brown'), ('size', 'small')): 13, 
                            (('shape', 'cube'), ('material', 'rubber'), ('color', 'cyan'), ('size', 'small')): 14, 
                            (('shape', 'cube'), ('material', 'rubber'), ('color', 'purple'), ('size', 'small')): 15, 
                            (('shape', 'cube'), ('material', 'rubber'), ('color', 'yellow'), ('size', 'small')): 16, 
                            (('shape', 'sphere'), ('material', 'metal'), ('color', 'gray'), ('size', 'small')): 17, 
                            (('shape', 'sphere'), ('material', 'metal'), ('color', 'red'), ('size', 'small')): 18, 
                            (('shape', 'sphere'), ('material', 'metal'), ('color', 'blue'), ('size', 'small')): 19, 
                            (('shape', 'sphere'), ('material', 'metal'), ('color', 'green'), ('size', 'small')): 20, 
                            (('shape', 'sphere'), ('material', 'metal'), ('color', 'brown'), ('size', 'small')): 21, 
                            (('shape', 'sphere'), ('material', 'metal'), ('color', 'cyan'), ('size', 'small')): 22, 
                            (('shape', 'sphere'), ('material', 'metal'), ('color', 'purple'), ('size', 'small')): 23, 
                            (('shape', 'sphere'), ('material', 'metal'), ('color', 'yellow'), ('size', 'small')): 24, 
                            (('shape', 'sphere'), ('material', 'rubber'), ('color', 'gray'), ('size', 'small')): 25, 
                            (('shape', 'sphere'), ('material', 'rubber'), ('color', 'red'), ('size', 'small')): 26, 
                            (('shape', 'sphere'), ('material', 'rubber'), ('color', 'blue'), ('size', 'small')): 27, 
                            (('shape', 'sphere'), ('material', 'rubber'), ('color', 'green'), ('size', 'small')): 28, 
                            (('shape', 'sphere'), ('material', 'rubber'), ('color', 'brown'), ('size', 'small')): 29, 
                            (('shape', 'sphere'), ('material', 'rubber'), ('color', 'cyan'), ('size', 'small')): 30, 
                            (('shape', 'sphere'), ('material', 'rubber'), ('color', 'purple'), ('size', 'small')): 31, 
                            (('shape', 'sphere'), ('material', 'rubber'), ('color', 'yellow'), ('size', 'small')): 32, 
                            (('shape', 'cylinder'), ('material', 'metal'), ('color', 'gray'), ('size', 'small')): 33, 
                            (('shape', 'cylinder'), ('material', 'metal'), ('color', 'red'), ('size', 'small')): 34, 
                            (('shape', 'cylinder'), ('material', 'metal'), ('color', 'blue'), ('size', 'small')): 35, 
                            (('shape', 'cylinder'), ('material', 'metal'), ('color', 'green'), ('size', 'small')): 36, 
                            (('shape', 'cylinder'), ('material', 'metal'), ('color', 'brown'), ('size', 'small')): 37, 
                            (('shape', 'cylinder'), ('material', 'metal'), ('color', 'cyan'), ('size', 'small')): 38, 
                            (('shape', 'cylinder'), ('material', 'metal'), ('color', 'purple'), ('size', 'small')): 39, 
                            (('shape', 'cylinder'), ('material', 'metal'), ('color', 'yellow'), ('size', 'small')): 40, 
                            (('shape', 'cylinder'), ('material', 'rubber'), ('color', 'gray'), ('size', 'small')): 41, 
                            (('shape', 'cylinder'), ('material', 'rubber'), ('color', 'red'), ('size', 'small')): 42, 
                            (('shape', 'cylinder'), ('material', 'rubber'), ('color', 'blue'), ('size', 'small')): 43, 
                            (('shape', 'cylinder'), ('material', 'rubber'), ('color', 'green'), ('size', 'small')): 44, 
                            (('shape', 'cylinder'), ('material', 'rubber'), ('color', 'brown'), ('size', 'small')): 45, 
                            (('shape', 'cylinder'), ('material', 'rubber'), ('color', 'cyan'), ('size', 'small')): 46, 
                            (('shape', 'cylinder'), ('material', 'rubber'), ('color', 'purple'), ('size', 'small')): 47, 
                            (('shape', 'cylinder'), ('material', 'rubber'), ('color', 'yellow'), ('size', 'small')): 48
                            }
    return code_characteristics

def reverse_lookup(reverse_dict, shape_idx, material_idx, color_idx):
    shapes = ['cube', 'sphere', 'cylinder']
    materials = ['metal', 'rubber']
    colors = ['gray', 'red', 'blue', 'green', 'brown', 'cyan', 'purple', 'yellow']
    sizes = ['small']  # Only one size

    # Create the search key from provided indices
    search_key = (
        ('shape', shapes[shape_idx]),
        ('material', materials[material_idx]),
        ('color', colors[color_idx]),
        ('size', sizes[0])
    )

    # Perform the lookup in the reverse dictionary
    class_id = reverse_dict[search_key]
    
    return class_id

def convert_from_multi_hot(multi_hot, zero_class_threshold = 0.025):
    batch_size, num_frames, attribute_channels, height, width = multi_hot.shape
    num_shapes = 3
    num_materials = 2
    num_colors = 8
    
    reverse_dict = getReverseDict()

    original_classes = torch.zeros(batch_size, num_frames, 1, height, width, device=multi_hot.device)

    # Process each attribute channel
    shape_indices = torch.argmax(multi_hot[:, :, :num_shapes, :, :], dim=2, keepdim=True)
    material_indices = torch.argmax(multi_hot[:, :, num_shapes:num_shapes + num_materials, :, :], dim=2, keepdim=True)
    color_indices = torch.argmax(multi_hot[:, :, num_shapes + num_materials:num_shapes + num_materials + num_colors, :, :], dim=2, keepdim=True)
    background_indicators = multi_hot[:, :, num_shapes + num_materials + num_colors, :, :]

    # Iterate through all elements to set class labels
    for b in tqdm(range(batch_size), leave=False):
        for f in range(num_frames):
            for y in range(height):
                for x in range(width):
                    # Check if all values are below the threshold
                    if background_indicators[b, f, y, x] > zero_class_threshold:
                        original_classes[b, f, 0, y, x] = 0
                    else:
                        shape = shape_indices[b, f, 0, y, x].item()
                        material = material_indices[b, f, 0, y, x].item()
                        color = color_indices[b, f, 0, y, x].item()
                        
                        original_classes[b, f, 0, y, x] = reverse_lookup(reverse_dict, shape, material, color)

    return original_classes.cpu().detach().numpy()