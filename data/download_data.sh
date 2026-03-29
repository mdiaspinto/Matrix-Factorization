#!/bin/bash
# Download MovieLens datasets for the eALS project
# Usage: ./download_data.sh [100k|1m|10m]

DATASET=${1:-100k}
DATA_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$DATASET" in
    100k)
        URL="https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        ZIP_FILE="ml-100k.zip"
        ;;
    1m)
        URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        ZIP_FILE="ml-1m.zip"
        ;;
    10m)
        URL="https://files.grouplens.org/datasets/movielens/ml-10m.zip"
        ZIP_FILE="ml-10m.zip"
        ;;
    *)
        echo "Usage: $0 [100k|1m|10m]"
        exit 1
        ;;
esac

echo "Downloading MovieLens $DATASET..."
cd "$DATA_DIR"
curl -LO "$URL"
unzip -o "$ZIP_FILE"
rm "$ZIP_FILE"
echo "Done. Data extracted to $DATA_DIR"
