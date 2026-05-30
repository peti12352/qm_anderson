#!/bin/bash
# Push qm_anderson to HPC
# Usage: ./scripts/hpc_push.sh

HPC_DEST="v@100.83.159.49:~/proj/petya/simulations/qm_anderson"
export SSHPASS="workwasmurder"

echo "Pushing codebase to $HPC_DEST..."
sshpass -e rsync -avz -e "ssh -o StrictHostKeyChecking=no" \
    --exclude 'data/' \
    --exclude '.pixi/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pdf' \
    --exclude '.DS_Store' \
    ./ "$HPC_DEST/"

echo "Push complete. (Note: data/ and .pixi/ were excluded to prevent overwriting HPC state)"
