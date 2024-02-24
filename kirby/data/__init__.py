from .data import ArrayDict, Data, Interval, IrregularTimeSeries, RegularTimeSeries
from .dataset import Dataset

from . import dandi_utils
from .dataset_builder import DatasetBuilder

from . import sampler
from .collate import collate, pad, track_mask, pad8, track_mask8, chain, track_batch



from kirby.taxonomy import Dictable, RecordingTech, StringIntEnum
from dataclasses import dataclass

class Hemisphere(StringIntEnum):
    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class Channel(Dictable):
    """Channels are the physical channels used to record the data. Channels are grouped
    into probes."""

    id: str
    local_index: int

    # Position relative to the reference location of the probe, in microns.
    relative_x_um: float
    relative_y_um: float
    relative_z_um: float

    area: StringIntEnum
    hemisphere: Hemisphere = Hemisphere.UNKNOWN


@dataclass
class Probe(Dictable):
    """Probes are the physical probes used to record the data."""

    id: str
    type: RecordingTech
    lfp_sampling_rate: float
    wideband_sampling_rate: float
    waveform_sampling_rate: float
    waveform_samples: int
    channels: list[Channel]
    ecog_sampling_rate: float = 0.0
