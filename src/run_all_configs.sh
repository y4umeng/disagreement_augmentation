#!/bin/bash

# Check if the directory path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/directory"
  exit 1
fi

# Get the directory path from the first argument
directory="$1"

# Check if the given path is a directory
if [ ! -d "$directory" ]; then
  echo "Error: $directory is not a valid directory."
  exit 1
fi

# Iterate over each file in the directory and execute a command
for file in "$directory"/*; do
  if [ -f "$file" ]; then
    # Replace 'your_command' with the actual command you want to execute
    python tools/train.py --cfg "$file"
  fi
done
