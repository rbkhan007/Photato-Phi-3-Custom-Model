#!/bin/bash
# Setup Ollama and import custom Phi-3 Mini model
# Usage: bash setup_ollama.sh path/to/model.gguf [model-name]

set -e

MODEL_FILE="${1:?Usage: $0 <model.gguf> [model-name]}"
MODEL_NAME="${2:-phi3-mini-custom}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phi-3 Mini Custom Model Setup ==="
echo ""

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        MINGW*|MSYS*|CYGWIN*) echo "windows";;
        *)          echo "unknown";;
    esac
}

OS=$(detect_os)
echo "Detected OS: $OS"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "Ollama not found. Installing..."
    echo ""

    case $OS in
        linux|macos)
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        windows)
            echo "Please install Ollama from: https://ollama.com/download/windows"
            echo "Then re-run this script."
            exit 1
            ;;
        *)
            echo "Unsupported OS. Please install Ollama manually from https://ollama.com"
            exit 1
            ;;
    esac
else
    echo "Ollama is already installed."
fi

# Check if model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo ""
    echo "Error: Model file not found: $MODEL_FILE"
    echo "Please provide a valid path to your quantized GGUF file."
    exit 1
fi

echo ""
echo "Model file: $MODEL_FILE"
echo "Model name: $MODEL_NAME"
echo ""

# Create Modelfile
TEMP_MODFILE=$(mktemp)
cat > "$TEMP_MODFILE" << EOF
FROM $MODEL_FILE

TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
"""

SYSTEM "You are a helpful, harmless, and honest assistant."

PARAMETER stop "<|end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
EOF

echo "Creating Ollama model..."
ollama create "$MODEL_NAME" -f "$TEMP_MODFILE"

# Clean up
rm -f "$TEMP_MODFILE"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To run your model:"
echo "  ollama run $MODEL_NAME"
echo ""
echo "To use in code:"
echo "  import ollama"
echo "  response = ollama.chat('$MODEL_NAME', messages=[{'role': 'user', 'content': 'Hello!'}])"
echo ""
echo "To list all models:"
echo "  ollama list"
