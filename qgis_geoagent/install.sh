#!/bin/bash
# Installation script for OpenGeoAgent
#
# Usage:
#   ./install.sh              # Install the plugin
#   ./install.sh --remove     # Remove the plugin
#   ./install.sh --name foo   # Install with custom name

set -e

# Default values
PLUGIN_NAME="open_geoagent"
REMOVE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --remove|-r)
            REMOVE=true
            shift
            ;;
        --name|-n)
            PLUGIN_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --remove, -r      Remove the plugin instead of installing"
            echo "  --name, -n NAME   Plugin folder name (default: open_geoagent)"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Detect QGIS plugin directory based on OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLUGIN_DIR="$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLUGIN_DIR="$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLUGIN_DIR="$APPDATA/QGIS/QGIS3/profiles/default/python/plugins"
else
    echo "Unknown OS type: $OSTYPE"
    echo "Please manually copy the open_geoagent folder to your QGIS plugins directory."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/open_geoagent"

echo "Platform: $OSTYPE"
echo "Plugin directory: $PLUGIN_DIR"
echo "Plugin name: $PLUGIN_NAME"
echo ""

if [[ "$REMOVE" == true ]]; then
    # Remove plugin
    TARGET_DIR="${PLUGIN_DIR}/${PLUGIN_NAME}"
    if [[ -d "$TARGET_DIR" ]]; then
        echo "Removing plugin: $TARGET_DIR"
        rm -rf "$TARGET_DIR"
        echo "Plugin removed successfully."
    else
        echo "Plugin not found. Nothing to remove."
    fi
else
    # Install plugin

    # Check source exists
    if [[ ! -d "$SOURCE_DIR" ]]; then
        echo "Error: Source directory not found: $SOURCE_DIR"
        exit 1
    fi

    # Create plugin directory if it doesn't exist
    mkdir -p "$PLUGIN_DIR"

    # Remove existing installation
    TARGET_DIR="${PLUGIN_DIR}/${PLUGIN_NAME}"
    if [[ -d "$TARGET_DIR" ]]; then
        echo "Removing existing installation..."
        rm -rf "$TARGET_DIR"
    fi

    # Copy plugin
    echo "Installing plugin to: $PLUGIN_DIR"
    cp -r "$SOURCE_DIR" "$TARGET_DIR"

    echo ""
    echo "============================================================"
    echo "Installation complete!"
    echo "============================================================"
    echo ""
    echo "To use the plugin:"
    echo "  1. Restart QGIS"
    echo "  2. Go to Plugins -> Manage and Install Plugins..."
    echo "  3. Enable '${PLUGIN_NAME}'"
    echo ""
fi
