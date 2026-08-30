"""
config.py
Configuration for AI Image Detector - GenImage (Tiny-GenImage) phase.
"""

import torch

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
HF_DATASET_NAME = "TheKernel01/Tiny-GenImage"

# Generator label mapping (from dataset card)
GENERATOR_ID_TO_NAME = {
    0: "Real",
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    5: "SD14",
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}
GENERATOR_NAME_TO_ID = {v: k for k, v in GENERATOR_ID_TO_NAME.items()}

# Generators held out ENTIRELY from training (never seen during train/val).
# Used only at final cross-generator evaluation time.
HELD_OUT_GENERATORS = ["Midjourney", "VQDM"]
HELD_OUT_GENERATOR_IDS = [GENERATOR_NAME_TO_ID[g] for g in HELD_OUT_GENERATORS]

# The remaining 6 generators (SEEN during training)
SEEN_GENERATOR_IDS = [
    gid for gid in GENERATOR_ID_TO_NAME.keys()
    if gid != 0 and gid not in HELD_OUT_GENERATOR_IDS
]

# ---------------------------------------------------------------------
# Split ratios
# ---------------------------------------------------------------------
# Applied independently to (a) each of the 6 seen generators' fake images
# and (b) the real images.
# train: used for training only
# val: used for in-training validation / checkpoint selection
# test_indist: held-out slice of the SEEN generators -> "in-distribution" test
# test_crossgen: held-out slice reserved for pairing with the fully-unseen
#                generators (Midjourney/VQDM) at cross-generator eval time
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_INDIST_RATIO = 0.10
TEST_CROSSGEN_RATIO = 0.10  # real-image slice only; held-out generators are 100% test

# ---------------------------------------------------------------------
# Multiclass (generator-ID) split ratios — separate task from the binary
# detector above. No held-out generator concept here: this model's whole
# job is to recognize all 8 generators by name, so all must be seen in train.
# ---------------------------------------------------------------------
MULTICLASS_TRAIN_RATIO = 0.70
MULTICLASS_VAL_RATIO = 0.10
MULTICLASS_TEST_RATIO = 0.20

RANDOM_SEED = 42

# ---------------------------------------------------------------------
# Image / model
# ---------------------------------------------------------------------
IMAGE_SIZE = 224  # required for pretrained ImageNet ResNet backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BACKBONE = "resnet18"  # swap to "resnet50" later as a controlled second experiment
FREEZE_BACKBONE = False  # freeze backbone weights, train only the binary head
FREEZE_LAYERS = ["conv1", "bn1", "layer1", "layer2"]
NUM_CLASSES = 2  # real vs fake (binary head)
NUM_CLASSES_MULTICLASS = 9  # real + 8 generators (generator-ID head)

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 11
MULTICLASS_NUM_EPOCHS = 15  # separate from binary NUM_EPOCHS -- 9-way task, more to learn
LEARNING_RATE_HEAD = 3e-4
LEARNING_RATE_BACKBONE = 1e-5
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/resnet_genimage_best.pt"
MULTICLASS_CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/resnet_generator_id_best.pt"
