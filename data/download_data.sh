#!/bin/bash
# Download and extract the Yelp dataset for the eALS project.
#
# NOTE: Yelp requires accepting their Terms of Service before downloading.
# Unlike MovieLens, the dataset cannot be fetched automatically.
#
# Step 1: Go to https://www.yelp.com/dataset
# Step 2: Click "Download Dataset" and accept the terms.
# Step 3: Save the downloaded file as "yelp_dataset.tar" in this directory.
# Step 4: Run this script: ./download_data.sh
#
# The script will extract the archive and place the relevant JSON file
# (yelp_academic_dataset_review.json) under data/yelp/.

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
TAR_FILE="$DATA_DIR/yelp_dataset.tar"
OUT_DIR="$DATA_DIR/yelp"

# Check the tar file is present
if [ ! -f "$TAR_FILE" ]; then
    echo ""
    echo "ERROR: yelp_dataset.tar not found in $DATA_DIR"
    echo ""
    echo "Yelp requires manual download:"
    echo "  1. Go to https://www.yelp.com/dataset"
    echo "  2. Click 'Download Dataset' and accept the Terms of Service."
    echo "  3. Save the file as: $TAR_FILE"
    echo "  4. Re-run this script."
    echo ""
    exit 1
fi

echo "Extracting Yelp dataset..."
mkdir -p "$OUT_DIR"
tar -xf "$TAR_FILE" -C "$OUT_DIR" yelp_academic_dataset_review.json

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Extraction failed."
    echo "Make sure yelp_dataset.tar is the full Yelp dataset archive."
    exit 1
fi

echo ""
echo "Done. Review data extracted to:"
echo "  $OUT_DIR/yelp_academic_dataset_review.json"
echo ""
echo "Next step: update data_loader.py to load from this path."
