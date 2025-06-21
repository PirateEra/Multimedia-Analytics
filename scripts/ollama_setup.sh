#!/bin/bash

set -e

# === CONFIG ===
OLLAMA_URL="https://ollama.com/download/ollama-linux-amd64.tgz"
OLLAMA_DIR="$HOME/ollama"
OLLAMA_BIN="$OLLAMA_DIR/ollama"
PROFILE_FILE="$HOME/.bashrc"  # or .zshrc if you're using zsh

# === Download Ollama binary ===
echo "Downloading Ollama..."
mkdir -p "$OLLAMA_DIR"
curl -L "$OLLAMA_URL" -o ollama-linux-amd64.tgz

# === Extract the binary ===
echo "Extracting to $OLLAMA_DIR..."
tar -xzf ollama-linux-amd64.tgz -C "$OLLAMA_DIR"

# === Add to PATH if not already there ===
if ! grep -q "$OLLAMA_DIR" "$PROFILE_FILE"; then
  echo "🔧 Adding Ollama to PATH in $PROFILE_FILE..."
  echo "export PATH=\"$OLLAMA_DIR:\$PATH\"" >> "$PROFILE_FILE"
  echo "Please run 'source $PROFILE_FILE' or restart your shell."
else
  echo "Ollama already in PATH."
fi

# === Try running ollama --version ===
echo "Verifying install..."
source "$PROFILE_FILE"  # Or manually do this later
"$OLLAMA_BIN" --version

# === Optionally start the server ===
echo "Starting Ollama server..."
nohup "$OLLAMA_BIN" serve > "$OLLAMA_DIR/ollama.log" 2>&1 &

echo "Ollama setup complete. Server running in background."
