#!/bin/bash
# Package Ariadne Datagen for Remote Pod

echo "📦 Packaging Ariadne Datagen files..."

ZIP_NAME="ariadne_datagen.zip"
rm -f $ZIP_NAME

zip -r $ZIP_NAME \
    src/datagen/generate_sft_dataset.py \
    src/datagen/ariadne.py \
    src/datagen/multilingual_corpora.py \
    src/datagen/llm_client.py \
    src/datagen/datadesigner_config.py

echo "✅ Created $ZIP_NAME"
echo "To upload: scp $ZIP_NAME user@pod-ip:~/"
