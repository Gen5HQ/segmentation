#!/bin/bash

mkdir -p models

if [ ! -f "models/sam_vit_b_01ec64.pth" ]; then
    echo "Downloading SAM ViT-B model..."
    curl -L -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    echo "Model downloaded successfully!"
else
    echo "Model already exists."
fi