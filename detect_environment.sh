#!/bin/bash

detect_environment() {
    # Check if the /cvmfs/config.mila.quebec/ directory exists
    if [ -d "/cvmfs/config.mila.quebec/" ]; then
        echo "mila"
        return
    fi

    # Get the hostname
    hostname=$(hostname)

    # Check if the hostname contains specific strings
    if [[ "$hostname" == *narval* ]]; then
        echo "narval"
    elif [[ "$hostname" == *beluga* ]]; then
        echo "beluga"
    elif [[ "$hostname" == bmedyer-gpu* ]]; then
        echo "gt"
    else
        echo "default"
    fi
}

detect_environment