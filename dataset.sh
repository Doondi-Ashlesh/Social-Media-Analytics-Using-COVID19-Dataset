#!/bin/bash

# --- Configuration ---
# Set the project directory name
PROJECT_DIR="twitter-covid-analysis"

# Determine the SCRATCH location
# Use $SCRATCH if set, otherwise default to /scratch/$USER
if [ -z "$SCRATCH" ]; then
    SCRATCH_PATH="/scratch/$USER"
else
    SCRATCH_PATH="$SCRATCH"
fi

DATA_DIR="$SCRATCH_PATH/$PROJECT_DIR/data"
ZIP_FILE="covid19-twitter-dataset.zip"
KAGGLE_URL="https://www.kaggle.com/api/v1/datasets/download/arunavakrchakraborty/covid19-twitter-dataset"

echo "--- Starting Dataset Setup ---"

# 1. Create the target directory in SCRATCH
echo "1. Creating project data directory at: $DATA_DIR"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# 2. Download the dataset using curl
echo "2. Downloading dataset from Kaggle..."
# -L follows redirects, -o specifies output file
curl -L -o "$ZIP_FILE" "$KAGGLE_URL"

# Check if the download was successful
if [ $? -eq 0 ] && [ -f "$ZIP_FILE" ]; then
    echo "   Download complete: $ZIP_FILE"
else
    echo "!!! Error: Download failed or zip file is missing."
    exit 1
fi

# 3. Unzip the file
echo "3. Unzipping data files..."
# Unzip creates a subdirectory named 'covid19-twitter-dataset'
unzip "$ZIP_FILE"

# Check if unzipping was successful
if [ $? -eq 0 ]; then
    echo "   Unzipping successful."
else
    echo "!!! Warning: Unzipping failed. Please check the zip file integrity."
fi

# 4. Clean up the zip file
echo "4. Removing temporary zip file: $ZIP_FILE"
rm "$ZIP_FILE"

echo "--- Dataset Setup Complete! ---"
echo "Your data is ready in the directory: $DATA_DIR"
echo "The CSV files are located in: $DATA_DIR/covid19-twitter-dataset/"
