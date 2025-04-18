import os
import h5py
import numpy as np
import shutil

DATASET_PATH = "./data_2024/augmented_data_2024/train"
AUGMENTATIONS = ["spatial_flip", "event_deletion", "temporal_shift"]
LABEL_SAMPLING_RATE = 100  # 100Hz (10ms per label step)

# Load train file names
def load_filenames(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]
TRAIN_FILES = load_filenames('./main/dataset/train_files.txt')

# Augmentation functions
def temporal_shift(events, labels, seed=42):
    """
    Apply a random temporal shift to events and adjust labels accordingly.
    
    Parameters:
    - events: NumPy structured array with fields ['t', 'x', 'y', 'p'], where 't' is in microseconds.
    - labels: NumPy array of shape (num_labels, label_dim), sampled at 100Hz (every 10ms).
    
    Returns:
    - events_shifted: Shifted events array.
    - labels_shifted: Adjusted labels array.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Step 1: Determine the range of event timestamps
    t_min = events['t'].min()
    t_max = events['t'].max()
    event_duration = t_max - t_min  # in microseconds (now around 4,000,000 us = 4 seconds)

    # Step 2: Define a reasonable shift range
    # Since events span 4 seconds, let's shift by up to ±200ms (200,000 us)
    max_shift_us = 200_000  # 200ms in microseconds
    shift_us = np.random.randint(-max_shift_us, max_shift_us + 1)  # Random shift in microseconds

    # Step 3: Shift the event timestamps
    events_shifted = events.copy()
    events_shifted['t'] = events_shifted['t'] + shift_us

    # Ensure timestamps don't go negative (optional, depending on your needs)
    if events_shifted['t'].min() < 0:
        events_shifted['t'] = events_shifted['t'] - events_shifted['t'].min()

    # Step 4: Adjust the labels
    # Labels are sampled at 100Hz, so each label corresponds to a 10ms window
    label_interval_us = 10_000  # 10ms in microseconds (100Hz)
    num_labels = len(labels)

    # Compute the new time window after shifting
    t_start_shifted = t_min + shift_us
    t_end_shifted = t_max + shift_us

    # Determine the label indices corresponding to the shifted time window
    # Label index for a timestamp t: floor(t / label_interval_us)
    label_start_idx = max(0, int(t_start_shifted // label_interval_us))
    label_end_idx = min(num_labels - 1, int(t_end_shifted // label_interval_us) + 1)

    # Extract the corresponding labels
    if label_start_idx <= label_end_idx:
        labels_shifted = labels[label_start_idx:label_end_idx + 1]
    else:
        # Handle case where indices are swapped (e.g., negative shift pushes events before t=0)
        labels_shifted = labels[label_end_idx:label_start_idx + 1]

    return events_shifted, labels_shifted

def spatial_flip(events, labels, width=640, height=480):
    events["x"] = width - 1 - events["x"]
    events["y"] = height - 1 - events["y"]
    labels[:, 0] = width - 1 - labels[:, 0]
    labels[:, 1] = height - 1 - labels[:, 1]
    return events, labels


def event_deletion(events, labels, delete_ratio=0.05):
    mask = np.random.rand(events.shape[0]) > delete_ratio
    return events[mask], labels

def augment(input_h5, output_h5, input_label, output_label, mode):
    # Read input .h5 file
    with h5py.File(input_h5, 'r') as f:
        events = f["events"][:].astype(np.dtype([("t", int), ("x", int), ("y", int), ("p", int)]))

    # Read input label file
    with open(input_label, "r") as f:
        labels = np.array([list(map(float, line.strip('()\n').split(', '))) for line in f.readlines()], np.float32)
            
    # Apply augmentation
    if mode == "temporal_shift":
        events, labels = temporal_shift(events, labels)
    if mode == "spatial_flip":
        events, labels = spatial_flip(events, labels)
    if mode == "event_deletion":
        events, labels = event_deletion(events, labels)
        
    # Write output .h5 file
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset("events", data=events, compression="gzip")

    # Write output label file
    augmented_labels = []
    for row in labels:
        x, y, close = row[0], row[1], row[2]
        augmented_labels.append(f"({int(x)}, {int(y)}, {int(close)})\n")
    with open(output_label, 'w') as f:
        f.writelines(augmented_labels)

def process_train_files(dataset_path, train_files):
    for folder in train_files:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            h5_file = os.path.join(folder_path, f"{folder}.h5")
            label_file = os.path.join(folder_path, "label.txt")

            if os.path.exists(h5_file) and os.path.exists(label_file):
                print(f"Processing {folder}...")

                for mode in AUGMENTATIONS:
                    aug_folder = os.path.join(dataset_path, f"{folder}_{mode}")
                    os.makedirs(aug_folder, exist_ok=True)

                    output_h5 = os.path.join(aug_folder, f"{folder}_{mode}.h5")
                    output_label = os.path.join(aug_folder, "label.txt")

                    augment(h5_file, output_h5, label_file, output_label, mode)

                    print(f"Saved {mode} augmented files to {aug_folder}")

# Run augmentation
process_train_files(DATASET_PATH, TRAIN_FILES)