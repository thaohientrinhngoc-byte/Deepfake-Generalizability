"""
Verification test for model initialization, forward pass, and DGS metric (Equation 1).
"""

import torch
from models import SpatiotemporalDeepfakeDetector
from utils import compute_dgs_score, evaluate_predictions


def test_models():
    print("Testing Spatiotemporal Models forward passes...")
    dummy_input = torch.randn(2, 30, 3, 224, 224)  # Batch=2, Frames=30, C=3, H=224, W=224

    for m_type in ["M_CNN_F", "M_CNN_T", "M_CL_F", "M_CL_T"]:
        model = SpatiotemporalDeepfakeDetector(model_type=m_type)
        output = model(dummy_input)
        assert output.shape == (2,), f"Output shape mismatch for {m_type}: expected (2,), got {output.shape}"
        print(f"  [SUCCESS] {m_type} forward pass OK -> output shape: {output.shape}")


def test_dgs():
    print("\nTesting DGS Score calculation against Paper Tables III–VI (Equation 1)...")
    # Table III (M_CNN-F on FF++): internal AUC = 76.47%, cross AUCs = [56.64%, 67.47%, 62.29%] -> DGS = 47.75%
    dgs_ff_cnn_f = compute_dgs_score(76.47, [56.64, 67.47, 62.29])
    assert round(dgs_ff_cnn_f, 2) == 47.75, f"Expected 47.75, got {dgs_ff_cnn_f:.2f}"
    print(f"  [SUCCESS] Table III (M_CNN-F, FF++) DGS_AUC: {dgs_ff_cnn_f:.2f}% (matches paper: 47.75%)")

    # Table VI (M_CL-T on DFDC): internal AUC = 99.48%, cross AUCs = [78.77%, 61.97%, 71.91%] -> DGS = 46.24%
    dgs_dfdc_cl_t = compute_dgs_score(99.48, [78.77, 61.97, 71.91])
    assert round(dgs_dfdc_cl_t, 2) == 46.24, f"Expected 46.24, got {dgs_dfdc_cl_t:.2f}"
    print(f"  [SUCCESS] Table VI  (M_CL-T, DFDC)  DGS_AUC: {dgs_dfdc_cl_t:.2f}% (matches paper: 46.24%)")

    # Table VI (M_CL-T on FF++): internal AUC = 91.38%, cross AUCs = [59.63%, 74.88%, 64.39%] -> DGS = 44.35%
    dgs_ff_cl_t = compute_dgs_score(91.38, [59.63, 74.88, 64.39])
    assert round(dgs_ff_cl_t, 2) == 44.35, f"Expected 44.35, got {dgs_ff_cl_t:.2f}"
    print(f"  [SUCCESS] Table VI  (M_CL-T, FF++)  DGS_AUC: {dgs_ff_cl_t:.2f}% (matches paper: 44.35%)")


if __name__ == "__main__":
    test_models()
    test_dgs()
    print("\nAll verification tests passed!")
