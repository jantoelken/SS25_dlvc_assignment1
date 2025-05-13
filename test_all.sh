#!/bin/zsh

# Directory containing the models
MODELS_DIR="/teamspace/studios/this_studio/SS25_dlvc_assignment1/saved_models"

# Path to your test script
TEST_SCRIPT="test_resnet18.py"

# Loop through all .pth files in the models directory
for MODEL_PATH in $MODELS_DIR/*.pth; do
    python $TEST_SCRIPT --path_to_trained_model $MODEL_PATH
done