# refer implementation of description_mpk to know how
# the simulated dataset is created
from test_dataset_sim import (
    description_mpk_allen,
    RUNNING_SPEED_MEAN,
    RUNNING_SPEED_STD,
    GABOR_POS_2D_MEAN,
    GABOR_POS_2D_STD,
)
from kirby.data import Dataset
from scripts.calculate_normalization_scales import calculate_zscales
from kirby.taxonomy import decoder_registry
from kirby.nn.multitask_readout import prepare_for_multitask_readout


def make_dataset(description_mpk, data_config=None):
    if not data_config:
        data_config = [
            {
                "selection": [
                    {
                        "dandiset": "allen_neuropixels_mock",
                    }
                ],
                "config": {
                    "multitask_readout": [
                        {
                            "decoder_id": "RUNNING_SPEED",
                            "normalize_mean": RUNNING_SPEED_MEAN,
                            "normalize_std": RUNNING_SPEED_STD,
                        },
                        {
                            "decoder_id": "GABOR_POS_2D",
                            "normalize_mean": [GABOR_POS_2D_MEAN, GABOR_POS_2D_MEAN],
                            "normalize_std": [GABOR_POS_2D_STD, GABOR_POS_2D_STD],
                        },
                    ],
                },
            }
        ]
    ds = Dataset(
        description_mpk,
        "train",
        data_config,
    )
    ds._check_for_data_leakage_flag = False

    return ds


def test_zscales_calculation(description_mpk_allen):
    ds = make_dataset(description_mpk_allen)
    zscales = calculate_zscales(ds)
    assert round(zscales["RUNNING_SPEED"][0]) == RUNNING_SPEED_MEAN
    assert round(zscales["RUNNING_SPEED"][1]) == RUNNING_SPEED_STD
    assert round(zscales["GABOR_POS_2D"][0][0]) == GABOR_POS_2D_MEAN
    assert round(zscales["GABOR_POS_2D"][0][1]) == GABOR_POS_2D_MEAN
    assert round(zscales["GABOR_POS_2D"][1][0]) == GABOR_POS_2D_STD
    assert round(zscales["GABOR_POS_2D"][1][1]) == GABOR_POS_2D_STD


def test_zscaling_in_multitask_readout(description_mpk_allen):
    ds = make_dataset(description_mpk_allen)
    data = ds.get("allen_neuropixels_mock/20100101_01", 0, 1)
    # prepare_for_multitask_readout should zscale the data if the config contains the mean and std
    _, _, values_dict, _, _ = prepare_for_multitask_readout(data, decoder_registry)

    for key, values in values_dict.items():
        # should be standard normal distribution
        assert round(values.mean()) == 0
        assert round(values.std()) == 1
