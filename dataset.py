"""
dataset.py
Downloads/loads Tiny-GenImage from HuggingFace and builds the splits needed
for cross-generator generalization testing:

  - train              : real + fake images from the 6 SEEN generators
  - val                : held-out slice of the same pool, used during training
  - test_indist        : separate held-out slice of the 6 SEEN generators
                          -> "have we learned these generators well"
  - test_crossgen      : real images (held-out slice) + Midjourney/VQDM fakes
                          (100% held out, never touched during training)
                          -> "do we generalize to unseen generators"

Run this file directly to sanity-check the splits:
    python dataset.py
"""

import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import io
from PIL import Image

import config


# ---------------------------------------------------------------------
# 1. Download / load
# ---------------------------------------------------------------------
def load_raw_dataset():
    """
    Downloads (or loads from local HF cache if already downloaded) the
    Tiny-GenImage dataset. 
    """
    print(f"Loading dataset: {config.HF_DATASET_NAME}")
    ds = load_dataset(config.HF_DATASET_NAME)
    print(ds)
    return ds


# ---------------------------------------------------------------------
# 2. Build seen / held-out index splits
# ---------------------------------------------------------------------
def _split_indices(indices, ratios, seed=config.RANDOM_SEED):
    """Shuffle and split a list of indices according to ratios dict."""
    rng = random.Random(seed)
    idx = list(indices)
    rng.shuffle(idx)

    n = len(idx)
    out = {}
    start = 0
    keys = list(ratios.keys())
    for i, key in enumerate(keys):
        if i == len(keys) - 1:
            out[key] = idx[start:]  # remainder, avoids rounding gaps
        else:
            count = int(n * ratios[key])
            out[key] = idx[start:start + count]
            start += count
    return out


def build_splits(hf_split):
    """
    hf_split: a single HF dataset split (e.g. concatenation of train+validation)
    Returns a dict of index lists: train, val, test_indist, test_crossgen
    """
    by_generator = defaultdict(list)
    for i, gen_id in enumerate(hf_split["generator"]):
        by_generator[gen_id].append(i)

    train_idx, val_idx, test_indist_idx, test_crossgen_idx = [], [], [], []

    # --- Real images (generator id 0): split 4 ways ---
    real_ratios = {
        "train": config.TRAIN_RATIO,
        "val": config.VAL_RATIO,
        "test_indist": config.TEST_INDIST_RATIO,
        "test_crossgen": config.TEST_CROSSGEN_RATIO,
    }
    real_splits = _split_indices(by_generator[0], real_ratios)
    train_idx += real_splits["train"]
    val_idx += real_splits["val"]
    test_indist_idx += real_splits["test_indist"]
    test_crossgen_idx += real_splits["test_crossgen"]

    # --- Seen generators' fake images: split 3 ways (no crossgen slice needed,
    #     they're not the unseen-generator test — that's what held-out gens are for) ---
    seen_ratios = {
        "train": config.TRAIN_RATIO,
        "val": config.VAL_RATIO,
        "test_indist": config.TEST_INDIST_RATIO + config.TEST_CROSSGEN_RATIO,
    }
    for gen_id in config.SEEN_GENERATOR_IDS:
        splits = _split_indices(by_generator[gen_id], seen_ratios)
        train_idx += splits["train"]
        val_idx += splits["val"]
        test_indist_idx += splits["test_indist"]

    # --- Held-out generators (Midjourney, VQDM): 100% goes to crossgen test ---
    for gen_id in config.HELD_OUT_GENERATOR_IDS:
        test_crossgen_idx += by_generator[gen_id]

    print("\n--- Split sizes ---")
    print(f"train:              {len(train_idx)}")
    print(f"val:                {len(val_idx)}")
    print(f"test_indist:        {len(test_indist_idx)}  (real + 6 seen generators)")
    print(f"test_crossgen:      {len(test_crossgen_idx)}  (real + Midjourney/VQDM, "
          f"never seen in training)")

    return {
        "train": train_idx,
        "val": val_idx,
        "test_indist": test_indist_idx,
        "test_crossgen": test_crossgen_idx,
    }


def build_multiclass_splits(hf_split, seed=config.RANDOM_SEED):
    """
    Splits for the SEPARATE generator-ID model (9-way: real + 8 generators).
    Unlike build_splits(), every generator is seen during training here --
    there's no held-out concept, because this model's job is to name the
    generator, not to test generalization to unseen ones.

    Real class is capped to roughly match the average fake-generator count,
    so training isn't dominated by the (usually much larger) real pool.
    """
    by_generator = defaultdict(list)
    for i, gen_id in enumerate(hf_split["generator"]):
        by_generator[gen_id].append(i)

    rng = random.Random(seed)

    all_gen_ids = sorted(config.GENERATOR_ID_TO_NAME.keys())  # 0..8
    fake_gen_ids = [g for g in all_gen_ids if g != 0]
    fake_counts = {g: len(by_generator[g]) for g in fake_gen_ids}
    avg_fake_count = int(sum(fake_counts.values()) / len(fake_counts))

    real_indices = list(by_generator[0])
    rng.shuffle(real_indices)

    real_count_before = len(real_indices)
    if real_count_before > avg_fake_count:
        real_indices = real_indices[:avg_fake_count]
    # If real is already smaller than avg_fake_count, we keep all of it --
    # capping only ever removes surplus, never fabricates more real images.
    by_generator[0] = real_indices

    ratios = {
        "train": config.MULTICLASS_TRAIN_RATIO,
        "val": config.MULTICLASS_VAL_RATIO,
        "test": config.MULTICLASS_TEST_RATIO,
    }

    train_idx, val_idx, test_idx = [], [], []
    print("\n--- Multiclass (generator-ID) split sizes ---")
    print(f"real: capped {real_count_before} -> {len(real_indices)} "
          f"(target ~= avg fake-generator count {avg_fake_count})")

    for gen_id in all_gen_ids:
        splits = _split_indices(by_generator[gen_id], ratios, seed=seed)
        train_idx += splits["train"]
        val_idx += splits["val"]
        test_idx += splits["test"]

        name = config.GENERATOR_ID_TO_NAME[gen_id]
        n = len(by_generator[gen_id])
        print(f"  class {gen_id} ({name:<12}): {n:>6} images "
              f"-> train {len(splits['train']):>5}, val {len(splits['val']):>4}, "
              f"test {len(splits['test']):>4}")

    print(f"\nTOTAL: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    if any(fake_counts[g] < avg_fake_count * 0.5 for g in fake_gen_ids):
        print("\nNote: some generators are well below the average count -- "
              "expect class imbalance among fakes too. Consider class-weighted "
              "loss in train_multiclass.py if per-class accuracy looks skewed.")

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

# ---------------------------------------------------------------------
# 3. PyTorch Dataset wrapper
# ---------------------------------------------------------------------
class RandomJPEGCompression:
    """Re-encode as JPEG at a random quality to simulate compression artifacts
    from different generators/platforms. Key for cross-generator generalization.
    Implemented as a class (not a lambda) so it's picklable for Windows
    multiprocessing (spawn-based DataLoader workers)."""

    def __init__(self, min_quality=10, max_quality=90, p=0.85):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(self.min_quality, self.max_quality)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

class RandomResolutionSimulation:
    """
    Diagnostic finding (see diagnose.py output): the model's %-predicted-fake
    on Midjourney (native 1024x1024, never seen at that scale during training
    -- seen generators top out at 512x512) is far lower than on seen
    generators, while real accuracy stays high. This suggests the model may
    be partly keying on resize/interpolation artifacts tied to specific
    native-resolution -> 224x224 downsampling ratios seen in training, which
    don't transfer to an unseen ratio like 1024 -> 224.
 
    This transform resizes the image to a random intermediate resolution
    (with a random interpolation kernel) BEFORE the final Resize(224,224),
    so the model sees a wide range of native-size -> 224 resampling chains
    during training -- including ratios beyond what any single seen
    generator's fixed native resolution would produce on its own.
 
    Note: this targets the resolution-mismatch signal specifically. It is
    not expected to fix the VQDM failure, since VQDM's native resolution
    (256x256) already matches ADM/GLIDE (both seen, both detected well) --
    that gap looks like a genuine artifact-signature difference (vector
    quantization vs. continuous diffusion), not a resolution issue.
 
    Implemented as a class (not a lambda) so it's picklable for Windows
    multiprocessing (spawn-based DataLoader workers).
    """
 
    def __init__(self, min_size=96, max_size=1280, p=0.7):
        self.min_size = min_size
        self.max_size = max_size
        self.p = p
        self.interp_modes = [
            Image.BILINEAR, Image.BICUBIC, Image.NEAREST, Image.LANCZOS,
        ]
 
    def __call__(self, img):
        if random.random() > self.p:
            return img
        target_short_side = random.randint(self.min_size, self.max_size)
        interp = random.choice(self.interp_modes)
        w, h = img.size
        if w <= h:
            new_w = target_short_side
            new_h = max(1, round(h * target_short_side / w))
        else:
            new_h = target_short_side
            new_w = max(1, round(w * target_short_side / h))
        return img.resize((new_w, new_h), interp)

 

def get_transforms(train: bool, task: str = "binary"):
    """
    task="binary": robustness augmentations (JPEG recompression, Gaussian blur)
        are applied on train, because the binary detector needs to generalize
        DESPITE compression/blur artifacts across unseen generators.
    task="multiclass": those same augmentations are deliberately OMITTED, even
        on train. The generator-ID model's whole job is to read compression/
        frequency fingerprints per generator -- stripping them out with JPEG
        recompression or blur would destroy the exact signal it needs to learn.
        Only resize + flip + normalize are applied.
    """
    if train and task == "binary":
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            RandomJPEGCompression(),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=7, sigma=(0.3, 2.5))], p=0.6
            ),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
    if train and task == "multiclass":
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
 

class GenImageDataset(Dataset):
    """
    Wraps a HF dataset split + a list of indices.
 
    task="binary" (default): returns (image_tensor, binary_label, generator_id).
        binary_label is 0=real/1=fake; generator_id is passed through only so
        eval.py can break accuracy down per generator.
    task="multiclass": returns (image_tensor, generator_id, generator_id).
        The target class IS the generator_id (0-8) here -- the label and the
        passthrough id are the same value, so downstream code (train/eval loops
        that unpack `images, labels, gen_ids`) works unchanged for both tasks.
    """
 
    def __init__(self, hf_split, indices, train: bool, task: str = "binary"):
        if task not in ("binary", "multiclass"):
            raise ValueError(f"Unknown task: {task}")
        self.hf_split = hf_split
        self.indices = indices
        self.task = task
        self.transform = get_transforms(train, task=task)
 
    def __len__(self):
        return len(self.indices)
 
    def __getitem__(self, i):
        row = self.hf_split[self.indices[i]]
        image = row["image"].convert("RGB")
        image = self.transform(image)
        generator_id = int(row["generator"])  # 0-8, see config.GENERATOR_ID_TO_NAME
 
        if self.task == "multiclass":
            label = generator_id               # target class = which generator (or real=0)
        else:
            label = int(row["label"])          # 0 = real, 1 = fake
 
        return image, label, generator_id
 

# ---------------------------------------------------------------------
# 4. Convenience builders
# ---------------------------------------------------------------------
def get_dataloaders():
    ds = load_raw_dataset()
    # Combine train+validation from HF into one pool, then re-split ourselves
    # so the split logic (seen vs held-out generators) is fully under our control.
    from datasets import concatenate_datasets
    full = concatenate_datasets([ds["train"], ds["validation"]])

    splits = build_splits(full)

    train_ds = GenImageDataset(full, splits["train"], train=True)
    val_ds = GenImageDataset(full, splits["val"], train=False)
    test_indist_ds = GenImageDataset(full, splits["test_indist"], train=False)
    test_crossgen_ds = GenImageDataset(full, splits["test_crossgen"], train=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)
    test_indist_loader = DataLoader(test_indist_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                                     num_workers=config.NUM_WORKERS, pin_memory=True)
    test_crossgen_loader = DataLoader(test_crossgen_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                                       num_workers=config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_indist_loader, test_crossgen_loader

def get_multiclass_dataloaders():
    """
    Builds loaders for the SEPARATE 9-class generator-ID model.
    Every generator (0=real .. 8=Wukong) appears in train/val/test -- no
    held-out concept here, unlike get_dataloaders(). Train-time transforms
    deliberately skip JPEG/blur augmentation (see get_transforms) so the
    model can still read generator-specific compression fingerprints.
    """
    ds = load_raw_dataset()
    from datasets import concatenate_datasets
    full = concatenate_datasets([ds["train"], ds["validation"]])
 
    splits = build_multiclass_splits(full)
 
    train_ds = GenImageDataset(full, splits["train"], train=True, task="multiclass")
    val_ds = GenImageDataset(full, splits["val"], train=False, task="multiclass")
    test_ds = GenImageDataset(full, splits["test"], train=False, task="multiclass")
 
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
 
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Sanity check: run `python dataset.py` to confirm splits look right
    # before kicking off a full training run.
    train_loader, val_loader, test_indist_loader, test_crossgen_loader = get_dataloaders()

    imgs, labels, gen_ids = next(iter(train_loader))
    print(f"\nSample train batch: images {imgs.shape}, labels {labels.shape}, "
          f"generator ids present: {sorted(set(gen_ids.tolist()))}")
