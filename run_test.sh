#!/bin/bash
# Quick test script for MiniMax-M2 Fused Operators
# This script demonstrates the usage and validates correctness

echo "======================================================================"
echo "MiniMax-M2 Fused Operators - Quick Test"
echo "======================================================================"
echo ""

# Check if PyTorch is available
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch is not installed. Please install PyTorch first."
    echo "Installation: pip install torch"
    exit 1
}

echo ""
echo "Running quick tests..."
echo "----------------------------------------------------------------------"

python3 test_minimax_m2_fused_ops.py --quick

echo ""
echo "======================================================================"
echo "Test completed successfully!"
echo "======================================================================"
echo ""
echo "To run full pytest suite:"
echo "  python3 -m pytest test_minimax_m2_fused_ops.py -v"
echo ""
echo "To run benchmark:"
echo "  python3 minimax_m2_fused_ops.py"
echo "======================================================================"
