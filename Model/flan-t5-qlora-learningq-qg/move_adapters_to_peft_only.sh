#!/bin/bash

MODEL_DIR="trained_model"
DEST_DIR="peft_only"

mkdir -p "$DEST_DIR"

cp "$MODEL_DIR/adapter_model.bin" "$DEST_DIR/"
cp "$MODEL_DIR/adapter_config.json" "$DEST_DIR/"

echo "PEFT adapter files copied to $DEST_DIR"