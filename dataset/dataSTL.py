import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm

pre_seq_length = 10
aft_seq_length = 20

def sample_frames(video_path, num_frames=20):
    # read the video
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # uniformly sample frames from the video
    frame_idxs = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = []
    for idx in frame_idxs:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        _, frame = video.read()
        # frame = cv2.resize(frame, (height, width))
        frames.append(frame)
    video.release()
    return np.stack(frames)

def extract_number(filename):
    if 'image' in filename:
        return int(filename.split('_')[1].split('.')[0])
    else:
        return float('inf')

def process_folder(folder_path, pre_slen=11, aft_slen=11):
    # get all the videos in this folder
    videos = []
    global_masks = []
    files = os.listdir(folder_path)
    
    for file in tqdm(files):
        broken = False
        video_path = os.path.join(folder_path, file)
        frames = []
        masks = []
        for frame in sorted(os.listdir(video_path), key=extract_number):
            if frame[-3:] == 'npy':
                mask = np.load(os.path.join(video_path, frame))
                
                masks.append(mask)
            else:
                image = cv2.imread(os.path.join(video_path, frame))
                if image is None:
                    broken = True
                frames.append(image)
                
        if np.shape(masks) != (1, 22, 160, 240) and folder_path != 'hidden':
            broken = True   
            
        if not broken:
            videos.append(np.stack(frames))
            if folder_path != 'hidden':
                global_masks.append(np.stack(masks))
        
    # stack video frames from each folder
    video_data = np.stack(videos).transpose(0, 1, 4, 2, 3)
    if folder_path != 'hidden':
        mask_data = np.stack(global_masks).transpose(0, 2, 1, 3, 4)
    else:
        mask_data = None

    # if the data is in [0, 255], rescale it into [0, 1]
    if video_data.max() > 1.0:
        video_data = video_data.astype(np.float32) / 255.0

    if mask_data is not None:
        return video_data[:, :pre_slen], video_data[:, aft_slen:], mask_data[:, :pre_slen], mask_data[:, aft_slen:]
    elif folder_path != 'hidden':
        return video_data[:, :pre_slen], video_data[:, aft_slen:], mask_data, mask_data
    else:
        return video_data, None, None, None

folders = ['train', 'val', 'hidden']
for folder in tqdm(folders):
    dataset = {}
    data_x, data_y, mask_x, mask_y = process_folder(folder, pre_slen=11, aft_slen=11)
    print(np.shape(data_x), np.shape(data_y), np.shape(mask_x), np.shape(mask_y))
    dataset['X_' + folder], dataset['Y_' + folder],  dataset['X_' + folder + '_mask'], dataset['Y_' + folder + '_mask'] = data_x, data_y, mask_x, mask_y
    with open(folder + '.pkl', 'wb') as f:
        pickle.dump(dataset, f)

# the shape is B x T x C x H x W
# B: the number of samples
# T: the number of frames in each sample
# C, H, W: the channel, height, width of each frame