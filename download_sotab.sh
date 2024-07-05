#!/bin/bash

# Function to download, extract, and delete a zip file from a URL
process_url() {
    local url=$1
    local dest_folder=$2
    local extract_folder=$3
    local file_name=$(basename "$url")
    local file_path="$dest_folder/$file_name"

    # Create the destination folder if it doesn't exist
    mkdir -p "$dest_folder"
    
    # Download the file
    echo "Downloading $url to $file_path"
    wget -q -O "$file_path" "$url"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        # Extract the zip file
        echo "Extracting $file_path to $extract_folder"
        unzip -qo "$file_path" -d "$extract_folder"

        # Delete the zip file
        echo "Deleting $file_path"
        rm -f "$file_path"
    else
        echo "Failed to download $url"
    fi
}

# Main script
urls=(
    "https://data.dws.informatik.uni-mannheim.de/structureddata/sotab/CTA_Training.zip"
    "https://data.dws.informatik.uni-mannheim.de/structureddata/sotab/CTA_Validation.zip"
    "https://data.dws.informatik.uni-mannheim.de/structureddata/sotab/CTA_Test.zip"
)
dest_folder="longsemannotator/data/raw_data"
extract_folder="longsemannotator/data/raw_data"

for url in "${urls[@]}"; do
    process_url "$url" "$dest_folder" "$extract_folder"
done

# Call the Python script
python3 preprocessing.py