#!/bin/bash
# Complete test suite for MiniMax-M2 Fused Operators with pytest
# Run this in a CUDA-enabled environment

echo "======================================================================"
echo "MiniMax-M2 Fused Operators - Full Test Suite"
echo "======================================================================"
echo ""

# Check dependencies
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'" || {
    echo "ERROR: CUDA-enabled PyTorch is required for these tests."
    exit 1
}

python3 -c "import pytest" || {
    echo "Installing pytest..."
    pip install pytest
}

echo "Starting test suite..."
echo "----------------------------------------------------------------------"
echo ""

# Run tests with verbose output
python3 -m pytest test_minimax_m2_fused_ops.py \
    -v \
    --tb=short \
    -s \
    --color=yes

TEST_EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "✗ Some tests failed (exit code: $TEST_EXIT_CODE)"
fi
echo "======================================================================"

exit $TEST_EXIT_CODE
