import os
import glob
import shutil
import random
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from skimage.transform import resize # Added for resizing
import matplotlib.pyplot as plt # Added for plotting

# --- To run from command line: ---
# python your_script_name.py --source_dirs "C:\Users\giles\Github\vesselFM\data\ImageCAS\1-200" "C:\Users\giles\Github\vesselFM\data\ImageCAS\201-400"`
# --output_base "C:\Users\giles\Github\vesselFM\data\d_real" --target_hw_dims 128 128
# To plot samples (after processing or if data exists):
# python your_script_name.py --output_base "C:\Users\giles\Github\vesselFM\data\d_real" --dataset_name "ImageCAS" --plot_samples
# (Note: --source_dirs and --target_hw_dims are not needed if only plotting existing data, but parser requires source_dirs)
# A better way for plotting only:
# python your_script_name.py --source_dirs DUMMY_PATH --output_base "C:\path\to\output" --plot_samples
# (Provide a dummy path for source_dirs if aggregate_and_split_imagecas is commented out or not run)

def aggregate_and_split_imagecas(
    source_dirs_list,
    output_base_dir,
    dataset_name="ImageCAS",
    split_ratios=(0.8, 0.1, 0.1),
    random_seed=42,
    target_hw_dims=(256, 256), # Changed: Target height and width
):
    """
    Aggregates ImageCAS data from multiple source directories, resizes HxW dimensions,
    and then splits it into train, validation, and test sets. Depth is preserved.

    Args:
        source_dirs_list (list): List of paths to directories containing
                                 original .img.nii.gz and .label.nii.gz files.
        output_base_dir (str): Base directory where aggregated and split data
                               will be stored. Each sample will include
                               img.npy, and mask.npy.
        dataset_name (str): Name for the dataset (e.g., "ImageCAS").
        split_ratios (tuple): Tuple of (train_ratio, val_ratio, test_ratio).
        random_seed (int): Seed for reproducible shuffling.
        target_hw_dims (tuple): Target dimensions (height, width) for resizing. Depth is preserved.
    """
    aggregated_dataset_path = Path(output_base_dir) / dataset_name
    split_dataset_path = Path(output_base_dir) / f"{dataset_name}-split"

    print(f"Creating aggregated dataset at: {aggregated_dataset_path}")
    os.makedirs(aggregated_dataset_path, exist_ok=True)

    aggregated_sample_paths = []
    global_sample_idx = 0

    for source_dir in source_dirs_list:
        print(f"Processing source directory: {source_dir}")
        img_files = sorted(glob.glob(os.path.join(source_dir, "*.img.nii.gz")))
        if not img_files:
            print(f"  No '*.img.nii.gz' files found in {source_dir}.")
            continue

        for img_file_path_str in img_files:
            img_file_path = Path(img_file_path_str)
            base_name = img_file_path.name.replace(".img.nii.gz", "")
            label_file_path = img_file_path.parent / f"{base_name}.label.nii.gz"

            if not label_file_path.exists():
                print(f"  Warning: Label file not found for {img_file_path}. Skipping.")
                continue

            sample_output_dir = aggregated_dataset_path / str(global_sample_idx)
            os.makedirs(sample_output_dir, exist_ok=True)

            try:
                # Load .nii.gz directly from source
                img_obj = nib.load(img_file_path)
                img_array = img_obj.get_fdata() # Original image data

                mask_obj = nib.load(label_file_path)
                mask_array = mask_obj.get_fdata() # Original mask data

                # print(f"    Original img shape: {img_array.shape}, mask shape: {mask_array.shape}") # Added print

                # Prepare target shape, preserving depth
                current_depth = img_array.shape[2]
                target_shape_for_resize = (target_hw_dims[0], target_hw_dims[1], current_depth)

                # Resize image
                # Ensure image is float before resize if it's not, to avoid issues with some skimage versions
                img_array_float = img_array.astype(np.float32)
                img_resized = resize(img_array_float,
                                     target_shape_for_resize,
                                     order=1,  # Bilinear interpolation
                                     preserve_range=True,
                                     anti_aliasing=True,
                                     mode='reflect')
                img_resized = img_resized.astype(np.float32) # Ensure float32 output

                # Resize mask
                mask_resized = resize(mask_array,
                                      target_shape_for_resize,
                                      order=0,  # Nearest-neighbor interpolation
                                      preserve_range=True, # Crucial for masks
                                      anti_aliasing=False, # No anti-aliasing for masks
                                      mode='reflect')
                mask_resized = mask_resized.astype(np.uint8) # Ensure mask is uint8

                # print(f"    Resized img shape: {img_resized.shape}, mask shape: {mask_resized.shape}") # Added print

                np.save(sample_output_dir / "img.npy", img_resized)
                np.save(sample_output_dir / "mask.npy", mask_resized)
                
                aggregated_sample_paths.append(sample_output_dir)
                # print(f"  Converted and resized {img_file_path.name} and {label_file_path.name} to .npy in {sample_output_dir}")
                global_sample_idx += 1
            except Exception as e:
                print(f"  Error processing, resizing, or converting files for {img_file_path.name}: {e}")

    print(f"\nAggregation complete. Total samples: {len(aggregated_sample_paths)}")

    if not aggregated_sample_paths:
        print("No samples aggregated. Skipping split.")
        return

    # --- Splitting data ---
    print(f"\nSplitting data into train/val/test sets at: {split_dataset_path}")
    random.seed(random_seed)
    random.shuffle(aggregated_sample_paths)

    n_total = len(aggregated_sample_paths)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    n_test = n_total - n_train - n_val  # Ensure all samples are used

    if n_train + n_val + n_test != n_total: # Should not happen with current logic
        print("Error in split calculation. Adjusting test set size.")
        n_test = n_total - n_train - n_val


    splits = {
        "train": aggregated_sample_paths[:n_train],
        "val": aggregated_sample_paths[n_train : n_train + n_val],
        "test": aggregated_sample_paths[n_train + n_val :],
    }

    for split_name, sample_paths_in_split in splits.items():
        current_split_dir = split_dataset_path / split_name
        os.makedirs(current_split_dir, exist_ok=True)
        print(f"  Creating {split_name} set with {len(sample_paths_in_split)} samples.")
        for src_sample_dir in sample_paths_in_split:
            sample_id = src_sample_dir.name
            # The destination for move should be the directory where the sample_id folder will reside
            dst_final_path_for_sample_dir = current_split_dir / sample_id
            try:
                if dst_final_path_for_sample_dir.exists():
                    shutil.rmtree(dst_final_path_for_sample_dir) # Clean up if exists from a previous run
                shutil.move(str(src_sample_dir), str(dst_final_path_for_sample_dir))
            except Exception as e:
                print(f"    Error moving sample {src_sample_dir} to {dst_final_path_for_sample_dir}: {e}")
        print(f"  Finished creating {split_name} set.")

    print("\nDataset aggregation and splitting complete.")
    print(f"  Total aggregated samples: {n_total}")
    print(f"  Train samples: {n_train} ({len(splits['train'])})")
    print(f"  Validation samples: {n_val} ({len(splits['val'])})")
    print(f"  Test samples: {n_test} ({len(splits['test'])})")


def plot_converted_samples(train_data_dir, num_to_plot=3):
    """
    Plots a few samples from the specified training data directory.

    Args:
        train_data_dir (Path): Path to the training data directory 
                               (e.g., .../ImageCAS-split/train).
        num_to_plot (int): Number of samples to plot.
    """
    print(f"\nPlotting up to {num_to_plot} training samples from: {train_data_dir}")
    if not train_data_dir.exists():
        print(f"  Training data directory not found: {train_data_dir}")
        return

    train_sample_dirs = sorted([d for d in train_data_dir.iterdir() if d.is_dir()])
    
    actual_num_to_plot = min(len(train_sample_dirs), num_to_plot)

    if actual_num_to_plot == 0:
        print("  No training samples found in the directory.")
        return

    for i in range(actual_num_to_plot):
        sample_dir = train_sample_dirs[i]
        try:
            img_array = np.load(sample_dir / "img.npy")
            mask_array = np.load(sample_dir / "mask.npy")

            # Get middle slice for plotting
            # Ensure there's a depth dimension, and it's the last one as per HWD format
            if img_array.ndim < 3 or mask_array.ndim < 3:
                print(f"  Skipping sample {sample_dir.name}: Data is not 3D.")
                continue
            
            depth_slice_idx = img_array.shape[2] // 2
            img_slice = img_array[:, :, depth_slice_idx]
            mask_slice = mask_array[:, :, depth_slice_idx]

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_slice, cmap="gray")
            axes[0].set_title(f"Image (Slice {depth_slice_idx})")
            axes[0].axis("off")

            axes[1].imshow(mask_slice, cmap="jet", alpha=0.5) 
            axes[1].set_title(f"Mask (Slice {depth_slice_idx})")
            axes[1].axis("off")
            
            plt.suptitle(f"Training Sample: {sample_dir.name}")
            plt.tight_layout()
            plt.show()
        except FileNotFoundError:
            print(f"  img.npy or mask.npy not found in {sample_dir}. Skipping.")
        except Exception as e:
            print(f"  Error plotting sample {sample_dir.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate and split ImageCAS dataset."
    )
    parser.add_argument(
        "--source_dirs",
        nargs="+",
        required=True,
        help="List of source directories containing ImageCAS NIfTI files.",
        # Example: --source_dirs C:\path\to\ImageCAS\1-200 C:\path\to\ImageCAS\201-400
    )
    parser.add_argument(
        "--output_base",
        type=str,
        required=True,
        help="Base output directory for 'ImageCAS' and 'ImageCAS-split' folders.",
        # Example: --output_base C:\Users\giles\Github\vesselFM\data\d_real
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ImageCAS",
        help="Name of the dataset (default: ImageCAS)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)."
    )
    parser.add_argument(
        "--target_hw_dims", # Renamed argument
        type=int,
        nargs=2, # Expects two integers: height and width
        default=[256, 256],
        help="Target dimensions for resizing (height width), e.g., 256 256. Depth is preserved. (default: 256 256)."
    )
    parser.add_argument(
        "--plot_samples",
        action="store_true",
        help="If set, plot a few training samples after processing or if data already exists.",
    )

    args = parser.parse_args()

    # --- Call the aggregation and splitting function ---
    # You can comment this out if data is already processed
    aggregate_and_split_imagecas(
        source_dirs_list=args.source_dirs,
        output_base_dir=args.output_base,
        dataset_name=args.dataset_name,
        random_seed=args.seed,
        target_hw_dims=tuple(args.target_hw_dims),
    )

    # --- Optionally plot samples ---
    if args.plot_samples:
        split_dataset_base_path = Path(args.output_base) / f"{args.dataset_name}-split"
        train_data_directory = split_dataset_base_path / "train"
        plot_converted_samples(train_data_directory, num_to_plot=3)

