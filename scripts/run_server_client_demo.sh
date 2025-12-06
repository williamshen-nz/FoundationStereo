#!/bin/bash
# Script to run FoundationStereo server and client in tmux

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Configuration
SESSION_NAME="foundation-stereo-demo"
CONDA_ENV="tiptop-foundation_stereo"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first:"
    echo "  sudo apt-get install tmux  # Ubuntu/Debian"
    echo "  sudo yum install tmux      # CentOS/RHEL"
    exit 1
fi

# Check if session already exists and kill it
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Killing old session..."
    tmux kill-session -t "$SESSION_NAME"
    sleep 1
fi

# Get conda initialization
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "Error: conda not found. Please ensure conda is installed and in PATH."
    exit 1
fi

# Create new tmux session with server
echo "Creating tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" -n "server-client"

# In the first pane, start the server
tmux send-keys -t "$SESSION_NAME:0.0" "source $CONDA_BASE/etc/profile.d/conda.sh" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "conda activate $CONDA_ENV" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "cd $SCRIPT_DIR/.." C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo 'Starting FoundationStereo server...'" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "python scripts/server.py" C-m

# Split window horizontally (creates pane on the right)
tmux split-window -h -t "$SESSION_NAME:0"

# In the second pane, wait for server to start, then run client with retries
tmux send-keys -t "$SESSION_NAME:0.1" "source $CONDA_BASE/etc/profile.d/conda.sh" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "conda activate $CONDA_ENV" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "cd $SCRIPT_DIR/.." C-m
tmux send-keys -t "$SESSION_NAME:0.1" "echo 'Waiting for server to initialize (this may take 10-20 seconds)...'" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "sleep 10" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "for i in 1 2 3; do echo \"Attempt \$i of 3: Running client example...\"; python scripts/client_example.py && break || { echo \"Connection failed, waiting 5 seconds before retry...\"; sleep 5; }; done" C-m

# Adjust pane sizes (50/50 split)
tmux select-layout -t "$SESSION_NAME:0" even-horizontal

# Attach to the session
echo ""
echo "============================================"
echo "Tmux session created successfully!"
echo "============================================"
echo ""
echo "Left pane:  Server"
echo "Right pane: Client"
echo ""
echo "Tmux commands:"
echo "  Ctrl+b then arrow keys - Switch between panes"
echo "  Ctrl+b then d          - Detach from session"
echo "  Ctrl+b then [          - Scroll mode (q to exit)"
echo "  Ctrl+c                 - Stop current process in active pane"
echo ""
echo "Note: Session will be killed automatically when you detach (Ctrl+b then d)"
echo ""
echo "Attaching to session..."
sleep 2
tmux attach-session -t "$SESSION_NAME"

# Kill session after detaching
echo ""
echo "Detached from session. Cleaning up..."
tmux kill-session -t "$SESSION_NAME" 2>/dev/null
echo "Session '$SESSION_NAME' has been terminated."
