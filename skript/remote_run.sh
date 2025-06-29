#!/bin/bash

set -e  # Stop on local errors

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <remote_host_alias>"
    exit 1
fi

REMOTE=$1

# 1. Local: checkout dev branch
echo "Switching to dev branch locally..."
git checkout dev

# 2. Local: call RSYNC (sourced command, assumed available in this environment)
echo "Running RSYNC to $REMOTE..."
RSYNC "$REMOTE"

# 3. Remote: prepare and build dev version
echo "Preparing remote dev build on $REMOTE..."
ssh "$REMOTE" bash << 'EOF'
cd $HOME/workspace/alpaka2_fork
rm -rf $HOME/workspace/alpaka2_fork/build/config
mkdir -p $HOME/workspace/alpaka2_fork/build/config
cd build
cmake --build . || echo "[Warning] CMake build failed — check if modules are loaded"
../skript/run_all.sh
EOF

# 4. Local: fetch results from dev (exhaustive)
echo "Fetching results from dev run..."
mkdir -p ../../RESULT/$REMOTE/exhaustive
rsync -avz "$REMOTE:~/workspace/alpaka2_fork/build/config/" "../../RESULT/$REMOTE/exhaustive"

# 5. Local: switch to dev.defaults
echo "Switching to dev.defaults branch locally..."
git checkout dev.defaults

# 6. Local: re-sync code
echo "Running RSYNC again to $REMOTE for dev.defaults..."
RSYNC "$REMOTE"

# 7. Remote: prepare and build dev.defaults version
echo "Preparing remote dev.defaults build on $REMOTE..."
ssh "$REMOTE" bash << 'EOF'
cd $HOME/workspace/alpaka2_fork
rm -rf $HOME/workspace/alpaka2_fork/build/config
mkdir -p $HOME/workspace/alpaka2_fork build/config
cd build
cmake --build . || echo "[Warning] CMake build failed — check if modules are loaded"
../skript/run_all.sh
EOF

# 8. Local: fetch results from default
echo "Fetching results from default run..."
mkdir -p ../../RESULT/$REMOTE/default
rsync -avz "$REMOTE:~/workspace/alpaka2_fork/build/config/" "../../RESULT/$REMOTE/default"

echo "✅ Done with all runs for $REMOTE"
