import pytest
import torch
import torchmetrics

from torch_brain.registry import MODALITIY_REGISTRY
from torch_brain.utils.stitcher import DecodingStitchEvaluator


@pytest.fixture
def mock_session_ids():
    return [
        "session1",
        "session2",
        "session3",
    ]


def test_initalization(mock_session_ids):

    # Test for R2Score
    evaluator = DecodingStitchEvaluator(
        session_ids=mock_session_ids,
        modality_spec=MODALITIY_REGISTRY["cursor_velocity_2d"],
    )
    assert list(evaluator.metrics.keys()) == mock_session_ids
    for metric in evaluator.metrics.values():
        assert isinstance(metric, torchmetrics.R2Score)

    # Test for Accuracy
    evaluator = DecodingStitchEvaluator(
        session_ids=mock_session_ids,
        modality_spec=MODALITIY_REGISTRY["drifting_gratings_orientation"],
    )
    assert list(evaluator.metrics.keys()) == mock_session_ids
    for metric in evaluator.metrics.values():
        assert isinstance(metric, torchmetrics.classification.MulticlassAccuracy)

    # Test custom metric factory
    metric_cls = torchmetrics.classification.BinaryAccuracy
    evaluator = DecodingStitchEvaluator(
        session_ids=mock_session_ids,
        metric_factory=metric_cls,
    )
    assert list(evaluator.metrics.keys()) == mock_session_ids
    for metric in evaluator.metrics.values():
        assert isinstance(metric, metric_cls)


def test_end_to_end_r2(mock_session_ids):
    B = 16  # batch size
    N = 32  # tokens per sample
    D = 2  # predictio dimension (for cursor velocity 2d)
    num_sessions = len(mock_session_ids)

    evaluator = DecodingStitchEvaluator(
        session_ids=mock_session_ids,
        modality_spec=MODALITIY_REGISTRY["cursor_velocity_2d"],
    )

    for epoch in range(3):
        assert len(evaluator._cache) == 0

        for batch_step in range(10):
            evaluator.update(
                timestamps=torch.linspace(0, 1, N).repeat(B, 1),
                preds=torch.rand(B, N, D),
                targets=torch.rand(B, N, D),
                eval_masks=torch.rand(B, N) > 0.5,
                session_ids=[
                    mock_session_ids[idx] for idx in torch.arange(B) % num_sessions
                ],
                absolute_starts=torch.rand(B),
            )

        metric_dict = evaluator.compute()
        assert len(metric_dict) == num_sessions

        evaluator.reset()


def test_end_to_end_accuracy(mock_session_ids):
    B = 9  # batch size
    N = 50  # tokens per sample
    D = 8  # predictio dimension (for drifting gratings orientation)
    num_sessions = len(mock_session_ids)

    evaluator = DecodingStitchEvaluator(
        session_ids=mock_session_ids,
        modality_spec=MODALITIY_REGISTRY["drifting_gratings_orientation"],
    )

    for epoch in range(3):
        assert len(evaluator._cache) == 0

        for batch_step in range(10):
            evaluator.update(
                timestamps=torch.linspace(0, 1, N).repeat(B, 1),
                preds=torch.rand(B, N, D),
                targets=torch.randint(D, (B, N, D)),
                eval_masks=torch.rand(B, N) > 0.5,
                session_ids=[
                    mock_session_ids[idx] for idx in torch.arange(B) % num_sessions
                ],
                absolute_starts=torch.rand(B),
            )

        metric_dict = evaluator.compute()
        assert len(metric_dict) == num_sessions

        evaluator.reset()
