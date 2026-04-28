#!/bin/bash
# Package the QGIS plugin for upload to the official QGIS plugin repository
#
# This script creates a zip file suitable for uploading to plugins.qgis.org
#
# Usage:
#   ./package_plugin.sh                          # Default packaging
#   ./package_plugin.sh --name my_plugin         # Custom plugin name
#   ./package_plugin.sh --output /path/to/out    # Custom output path
#
# Requirements:
#   - zip command

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
PLUGIN_NAME="open_geoagent"
SOURCE_DIR="${SCRIPT_DIR}/open_geoagent"
OUTPUT_DIR="${SCRIPT_DIR}"
INCLUDE_VERSION=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name|-n)
            PLUGIN_NAME="$2"
            shift 2
            ;;
        --source|-s)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-version)
            INCLUDE_VERSION=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --name, -n NAME       Plugin name for zip folder (default: open_geoagent)"
            echo "  --source, -s PATH     Source plugin directory"
            echo "  --output, -o PATH     Output directory for zip file"
            echo "  --no-version          Don't include version in filename"
            echo "  --help, -h            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if source directory exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Get version from metadata.txt
VERSION=$(grep "^version=" "${SOURCE_DIR}/metadata.txt" 2>/dev/null | cut -d= -f2 | tr -d '[:space:]')
if [[ -z "$VERSION" ]]; then
    VERSION="unknown"
fi

# Create output filename
if [[ "$INCLUDE_VERSION" == true ]]; then
    ZIP_NAME="${PLUGIN_NAME}-${VERSION}.zip"
else
    ZIP_NAME="${PLUGIN_NAME}.zip"
fi

OUTPUT_PATH="${OUTPUT_DIR}/${ZIP_NAME}"

echo "Packaging QGIS Plugin"
echo "====================="
echo "Source directory: $SOURCE_DIR"
echo "Plugin name: $PLUGIN_NAME"
echo "Version: $VERSION"
echo "Output: $OUTPUT_PATH"
echo ""

# Create temp directory for packaging
TEMP_DIR=$(mktemp -d)
PACKAGE_DIR="${TEMP_DIR}/${PLUGIN_NAME}"

# Copy plugin files
echo "Copying plugin files..."
mkdir -p "$PACKAGE_DIR"
cp -r "${SOURCE_DIR}/"* "$PACKAGE_DIR/"

# Remove unwanted files and directories
echo "Cleaning up..."
find "$PACKAGE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PACKAGE_DIR" -type d -name ".git" -exec rm -rf {} + 2>/dev/null || true
find "$PACKAGE_DIR" -type d -name ".svn" -exec rm -rf {} + 2>/dev/null || true
find "$PACKAGE_DIR" -type d -name ".idea" -exec rm -rf {} + 2>/dev/null || true
find "$PACKAGE_DIR" -type d -name ".vscode" -exec rm -rf {} + 2>/dev/null || true
find "$PACKAGE_DIR" -type d -name "__MACOSX" -exec rm -rf {} + 2>/dev/null || true
find "$PACKAGE_DIR" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find "$PACKAGE_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$PACKAGE_DIR" -type f -name "*.pyo" -delete 2>/dev/null || true
find "$PACKAGE_DIR" -type f -name "*.bak" -delete 2>/dev/null || true
find "$PACKAGE_DIR" -type f -name "*~" -delete 2>/dev/null || true
find "$PACKAGE_DIR" -type f -name ".DS_Store" -delete 2>/dev/null || true
find "$PACKAGE_DIR" -type f -name "ui_*.py" -delete 2>/dev/null || true
find "$PACKAGE_DIR" -type f -name "resources_rc.py" -delete 2>/dev/null || true

# Remove existing zip if it exists
if [[ -f "$OUTPUT_PATH" ]]; then
    rm "$OUTPUT_PATH"
fi

# Create zip file
echo "Creating zip archive..."
cd "$TEMP_DIR"
zip -r "$OUTPUT_PATH" "$PLUGIN_NAME" -x "*.pyc" -x "*__pycache__*" -x "*.git*"

# Cleanup
rm -rf "$TEMP_DIR"

# Display results
ZIP_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
echo ""
echo "=================================================="
echo "Plugin packaged successfully!"
echo "=================================================="
echo "Output: $OUTPUT_PATH"
echo "Size: $ZIP_SIZE"
echo ""
echo "To install locally, copy the zip contents to:"
echo "  Linux:   ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/"
echo "  macOS:   ~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/"
echo "  Windows: %APPDATA%/QGIS/QGIS3/profiles/default/python/plugins/"
echo ""
echo "To upload to QGIS Plugin Repository:"
echo "  https://plugins.qgis.org/plugins/"
echo "=================================================="

# Verify zip contents
echo ""
echo "Zip contents:"
echo "-------------"
unzip -l "$OUTPUT_PATH" | head -30
