#!/bin/bash
# Pull qm_anderson data from HPC
# Usage: ./scripts/hpc_pull.sh

HPC_SRC="v@100.83.159.49:~/proj/petya/simulations/qm_anderson"
export SSHPASS="workwasmurder"

echo "Pulling data/ from $HPC_SRC..."
sshpass -e rsync -avz -e "ssh -o StrictHostKeyChecking=no" "$HPC_SRC/data/" ./data/

echo "Pull complete."
