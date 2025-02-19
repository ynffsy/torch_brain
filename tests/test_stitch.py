import torch
from torch_brain.utils.stitcher import stitch


def test_stitch_float():
    in_timestamps = torch.tensor([0.1, 0.3, 0.42, 0.5, 0.3, 0.3, 0.5])

    # Test on a (N, D) shaped input
    D = 3
    in_values = torch.rand((len(in_timestamps), D))
    out_values = stitch(in_timestamps, in_values)
    expected_outputs = torch.stack(
        [
            in_values[0],
            (in_values[1] + in_values[4] + in_values[5]) / 3.0,
            in_values[2],
            (in_values[3] + in_values[6]) / 2.0,
        ]
    )
    assert torch.equal(out_values, expected_outputs)

    # Test on (N,) shaped input
    in_values = torch.rand(len(in_timestamps))
    out_values = stitch(in_timestamps, in_values)
    expected_outputs = torch.tensor(
        [
            in_values[0],
            (in_values[1] + in_values[4] + in_values[5]) / 3.0,
            in_values[2],
            (in_values[3] + in_values[6]) / 2.0,
        ]
    )
    print(in_values)
    print(out_values)
    print(expected_outputs)
    assert torch.equal(out_values, expected_outputs)


def test_stitch_long():
    in_timestamps = torch.tensor([0.1, 0.3, 0.42, 0.5, 0.3, 0.3, 0.5, 0.15, 0.15])
    in_values = torch.LongTensor([0, 1, 2, 3, 1, 2, 3, 6, 5])
    out_values = stitch(in_timestamps, in_values)
    expected_outputs = torch.LongTensor([0, 5, 1, 2, 3])
    assert torch.equal(out_values, expected_outputs)
