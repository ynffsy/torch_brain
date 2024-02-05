import pytest

import numpy as np
import torch

from kirby.data.sampler import SequentialFixedWindowSampler, RandomFixedWindowSampler
from kirby.data.dataset import DatasetIndex


# helper
def compare_slice_indices(a, b):
    return (
        (a.session_id == b.session_id) and
        np.isclose(a.start, b.start) and
        np.isclose(a.end, b.end)
    )

# helper
def samples_in_interval_dict(samples, interval_dict):
    for s in samples:
        assert s.session_id in interval_dict
        allowed_intervals = interval_dict[s.session_id]
        if not (
            sum([
                (s.start >= start) and (s.end <= end)
                for start, end in allowed_intervals
            ]) == 1
        ):
            return False

    return True
        

def test_sequential_sampler():
    sampler = SequentialFixedWindowSampler(
        interval_dict = {
            "session1": [(0., 2.), (3., 4.5),], # 3
            "session2": [(0.1, 1.25), (2.5, 5.), (15., 18.7)], # 7
            "session3": [(1000., 1002.),], #2
        },
        window_length=1.1,
        step=0.75,
    )
    assert len(sampler) == 18

    s_iter = iter(sampler)
    assert compare_slice_indices(next(s_iter), DatasetIndex("session1", 0., 1.1))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session1", 0.75, 1.85))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session1", 0.9, 2.))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session1", 3., 4.1))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session1", 3.4, 4.5))
    #
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 0.1, 1.2))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 0.15, 1.25))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 2.5, 3.6))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 3.25, 4.35))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 3.9, 5.))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 15., 16.1))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 15.75, 16.85))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 16.5, 17.6))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 17.25, 18.35))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session2", 17.6, 18.7))
    #
    assert compare_slice_indices(next(s_iter), DatasetIndex("session3", 1000., 1001.1))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session3", 1000.75, 1001.85))
    assert compare_slice_indices(next(s_iter), DatasetIndex("session3", 1000.9, 1002.))


def test_random_sampler():

    interval_dict = {
        "session1": [(0., 2.), (3., 4.5),], # 3
        "session2": [(0.1, 1.25), (2.5, 5.), (15., 18.7)], # 7
        "session3": [(1000., 1002.),], #2
    }

    sampler = RandomFixedWindowSampler(
        interval_dict=interval_dict,
        window_length=1.1,
        generator=torch.Generator().manual_seed(42),
    )
    assert len(sampler) == 9

    # sample and check that all indices are within the expected range
    samples = list(sampler)
    assert len(samples) == 9
    assert samples_in_interval_dict(samples, interval_dict) == True

    # sample again and check that the indices are different this time
    samples2 = list(sampler)
    assert len(samples) == 9
    for s1 in samples:
        for s2 in samples2:
            assert not compare_slice_indices(s1, s2)
    

    # Test "index in valid range" when step > window_length
    sampler = RandomFixedWindowSampler(
        interval_dict=interval_dict,
        window_length=1.1,
        generator=torch.Generator().manual_seed(42),
    )
    samples = list(sampler)
    assert samples_in_interval_dict(samples, interval_dict) == True


    # Having window_length bigger than any interval should raise an error
    with pytest.raises(AssertionError):
        sampler = RandomFixedWindowSampler(
            interval_dict=interval_dict,
            window_length=3,
            generator=torch.Generator().manual_seed(42),
        )

        [_ for _ in sampler]