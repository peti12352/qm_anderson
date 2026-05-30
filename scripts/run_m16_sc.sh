#!/bin/bash
# Run M=16 for SC Dilution
# Expected runtime: ~26 hours on HPC (RTX 5090 equivalent or strong CPU cluster depending on backend)
# This script should be pushed to the HPC and executed there (e.g., via slurm or nohup).

echo "Starting M=16 SC Dilution Run..."
echo "This will take approximately 26 hours to complete."

# Run the updated sc_dilution.py which now has M_LIST=[8, 10, 12, 16]
pixi run python runs/sc_dilution.py

echo "Run complete! You can now pull the data back locally."
