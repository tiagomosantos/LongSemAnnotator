# Navigate into the main project directory
cd longsemannotator

# Create the data and checkpoints directories
mkdir -p data checkpoints

# Navigate into the data directory
cd data

# Create the subdirectories inside the data directory
mkdir -p embeddings raw_data ready_to_model_data structured_data

# Print completion message
echo "Directory structure created successfully."