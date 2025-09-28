
#!/usr/bin/env bash
# Download helper for model weights.
set -euo pipefail
MODEL_URL="https://github.com/mdkarimuddin/covid-flask/releases/download/v1.0.0-fixed/model_fixed.keras"
DEST_PATH="model_fixed.keras"

if [ -f "$DEST_PATH" ]; then
  echo "Model file '$DEST_PATH' already exists. Skipping download."
  exit 0
fi

echo "Downloading model from $MODEL_URL ..."
curl -L -o "$DEST_PATH" "$MODEL_URL"
echo "Downloaded '$DEST_PATH'"

