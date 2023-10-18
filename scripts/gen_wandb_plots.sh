#!/bin/bash

# Check if the directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIRECTORY="$1"

# Check if the directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory $DIRECTORY does not exist."
    exit 1
fi

# Change to the specified directory
cd "$DIRECTORY"

# For each folder in the directory
for folder in */; do
    # Create a new folder with the _csv suffix
    new_folder="${folder%/}_csv"
    rm -rf "$new_folder"
    mkdir -p "$new_folder"
    echo $DIRECTORY
    # List all csv files in the current folder
    csv_files=$(find "${folder}" -type f -name "*.csv")

    # If there are no CSV files, the array will contain the pattern itself
    # So, we check the first item to see if it exists
#    if [ -f "${csv_files[0]}" ]; then
#        # Run the python script for each csv file
    echo $csv_files
    for csv_file in $csv_files; do
      echo "$DIRECTORY/$csv_file"
#      echo "$DIRECTORY/$new_folder"
        python /home/i53/student/jandl/repos/DIGNet/scripts/wandb2plt.py -f "$DIRECTORY/$csv_file" -p "$DIRECTORY/$new_folder"
    done
#    fi
done
