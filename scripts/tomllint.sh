#!/bin/bash

# set -x

status_code="0"
TAPLO_URL=https://github.com/tamasfe/taplo/releases/download/0.8.1

# Process commmand-line arguments
if [ $# -eq 0 ]; then
    TARGET=$(pwd)
elif [ $# -eq 1 ]; then
    TARGET="$1"
fi

check_platform() {
    # Enumerate platform and set binary name appropriately
    PLATFORM=$(uname -a)
    if (echo "${PLATFORM}" | grep Darwin | grep arm64); then
        TAPLO_BIN="taplo-darwin-aarch64"
    elif (echo "${PLATFORM}" | grep Darwin | grep x86_64); then
        TAPLO_BIN="taplo-darwin-x86_64"
    elif (echo "${PLATFORM}" | grep Linux | grep aarch64); then
        TAPLO_BIN="taplo-full-linux-aarch64"
    elif (echo "${PLATFORM}" | grep Linux | grep x86_64); then
        TAPLO_BIN="taplo-full-linux-x86_64"
    else
        echo "Unsupported platform!"; exit 1
    fi
    TAPLO_GZIP="$TAPLO_BIN.gz"

}

check_file() {
    local file_path="$1"
    cp "$file_path" "$file_path.original"
    /tmp/"${TAPLO_BIN}" format "$file_path" >/dev/null
    diff "$file_path" "$file_path.original"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        status_code=$exit_code
        echo "::error file={$file_path},line={line},col={col}::{TOML file not formatted}"
    elif [ -f "$file_path.original" ]; then
        rm "$file_path.original"
    fi
}

check_all() {
    if [ -d "${TARGET}" ]; then
        echo "Scanning all the TOML files at folder: ${TARGET}"
    fi
    while IFS= read -r current_file; do
        echo "Check file $current_file"
        check_file "$current_file"
    done < <(find . -name '*.toml' -type f -not -path '*/.*')
}

download_taplo() {
    if [ ! -f /tmp/"${TAPLO_GZIP}" ]; then
        "${WGET_BIN}" -q -e robots=off -P /tmp "${TAPLO_URL}"/"${TAPLO_GZIP}"
    fi
    TAPLO_PATH="/tmp/${TAPLO_BIN}"
    if [ ! -x "${TAPLO_PATH}" ]; then
        gzip -d "/tmp/${TAPLO_GZIP}"
        chmod +x "/tmp/${TAPLO_BIN}"
    fi
    TAPLO_BIN="/tmp/${TAPLO_BIN}"
}

cleanup_tmp() {
    # Only clean the temp directory if it was used
    if [ -f /tmp/"${TAPLO_BIN}" ] || [ -f /tmp/"${TAPLO_GZIP}" ]; then
        echo "Cleaning up..."
        rm /tmp/"${TAPLO_BIN}"*
    fi
}

check_wget() {
    # Pre-flight binary checks and download
    WGET_BIN=$(which wget)
    if [ ! -x "${WGET_BIN}" ]; then
        echo "WGET command not found"
        sudo apt update; sudo apt-get install -y wget | true
    fi
    WGET_BIN=$(which wget)
    if [ ! -x "${WGET_BIN}" ]; then
        echo "WGET could not be installed"; exit 1
    fi
}

TAPLO_BIN=$(which taplo)
if [ ! -x "${TAPLO_BIN}" ]; then
    check_wget && check_platform && download_taplo
fi

if [ ! -x "${TAPLO_BIN}" ]; then
    echo "Download failed: TOML linting binary not found [taplo]"
    status_code="1"
else
    # To avoid execution when sourcing this script for testing
    [ "$0" = "${BASH_SOURCE[0]}" ] && check_all "$@"
fi

cleanup_tmp
exit $status_code
