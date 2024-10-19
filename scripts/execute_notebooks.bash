#!/bin/bash

# Define an array of patterns to skip
SKIP_LIST=("autophase" "awa_" "devel_" "dipole" "movie")

# Stop on first error
set -e

export PYDEVD_DISABLE_FILE_VALIDATION=1

# Function to print colored text
print_color() {
    if [[ -t 1 && "$TERM" != "dumb" ]]; then  # Check if terminal supports colors
        case $1 in
            red)
                color_code=$(tput setaf 1)
                ;;
            green)
                color_code=$(tput setaf 2)
                ;;
            yellow)
                color_code=$(tput setaf 3)
                ;;
            blue)
                color_code=$(tput setaf 4)
                ;;
            reset)
                color_code=$(tput sgr0)
                ;;
            *)
                color_code=$(tput sgr0)
                ;;
        esac
        echo -e "${color_code}$2$(tput sgr0)"
    else
        echo "$2"  # No color in environments without valid terminal
    fi
}

NOTEBOOKS=$(find . -type f -name "*.ipynb" -not -path '*/.*')

echo $NOTEBOOKS

for file in $NOTEBOOKS
do
    start_time=$(date +%s)  # Start time in seconds

    SKIP_FILE=false
    for pattern in "${SKIP_LIST[@]}"; do
        if [[ "$file" == *"$pattern"* ]]; then
            SKIP_FILE=true
            break
        fi
    done

    if [ "$SKIP_FILE" = true ]; then
        print_color "yellow" "Skipping $file"
        continue
    fi

    print_color "blue" "Executing $file"

    # Run the notebook and stop on error
    if ! jupyter nbconvert --to notebook --execute "$file" --inplace; then
        print_color "red" "Error encountered while executing $file. Stopping."
        exit 1
    fi

    end_time=$(date +%s)  # End time in seconds
    elapsed=$((end_time - start_time))  # Calculate elapsed time in seconds
    print_color "green" "Execution time for $file: ${elapsed}s"
done
