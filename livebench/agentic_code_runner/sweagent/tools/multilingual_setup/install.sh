#!/usr/bin/env bash

# Define variables to exclude
EXCLUDE_VARS="PWD|LANG|PYTHONPATH|ROOT|PS0|PS1|PS2|_|OLDPWD|LC_ALL|LANG|LSCOLORS|SHLVL"


echo "Original Environment Variables:"
env | sort

# Only add Python 3.11 to PATH if no python exists
if ! command -v python &> /dev/null; then
    echo -e "\nNo Python found in system"
    
    # Only add /root/python3.11 if it exists
    if [ -d "/root/python3.11" ]; then
        echo "Adding Python 3.11 to PATH"
        export PATH="/root/python3.11/bin:$PATH"
    else
        echo "Directory /root/python3.11 does not exist, skipping PATH update"
    fi

    # Find the actual python3 path and create symlinks based on its location
    if PYTHON3_PATH=$(which python3 2>/dev/null); then
        PYTHON_DIR=$(dirname "$PYTHON3_PATH")
        echo "Found Python3 at: $PYTHON3_PATH"
        
        # Create python/pip aliases in the same directory as python3
        ln -sf "$PYTHON3_PATH" "$PYTHON_DIR/python"
        ln -sf "$PYTHON_DIR/pip3" "$PYTHON_DIR/pip"
        echo "Created symlinks: python -> python3, pip -> pip3 in $PYTHON_DIR"
    else
        echo "Could not find python3 executable"
    fi
else
    echo -e "\nPython already exists in system, skipping Python 3.11 setup"
fi

# Attempt to read and set process 1 environment
echo -e "\nSetting environment variables from /proc/1/environ..."
if [ -r "/proc/1/environ" ]; then
    while IFS= read -r -d '' var; do
        # Skip excluded variables
        if ! echo "$var" | grep -qE "^(${EXCLUDE_VARS})="; then
            # If the variable is PATH, append and deduplicate
            if [[ "$var" =~ ^PATH= ]]; then
                # Combine paths and remove duplicates while preserving order
                export PATH="$(echo "${PATH}:${var#PATH=}" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//')"
            else
                export "$var"
            fi
        fi
    done < /proc/1/environ
    echo "Successfully imported environment from /proc/1/environ"
else
    echo "Cannot access /proc/1/environ - Permission denied"
fi

# Print updated environment variables
echo -e "\nUpdated Environment Variables:"
env | sort
