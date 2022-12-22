##################################################################
# 2022 - King Abdullah University of Science and Technology (KAUST)
#
# Authors: Nick Luiken, Matteo Ravasi
# Description: Auxiliary routines for data handling in SS networks
##################################################################

import re
from enum import Enum, auto
from collections import OrderedDict
from typing import Dict, Any, List


def list_constants(clazz: Any, private: bool = False) -> List[Any]:
    """Fetch all values from variables formatted as constants in a class.

    Args:
        clazz (Any): Class to fetch constants from.

    Returns:
        List[Any]: List of values.
    """
    variables = [i for i in dir(clazz) if not callable(i)]
    regex = re.compile(r"^{}[A-Z0-9_]*$".format("" if private else "[A-Z]"))
    names = list(filter(regex.match, variables))
    values = [clazz.__dict__[name] for name in names]
    return values


class DataDim(Enum):
    BATCH = auto()
    CHANNEL = auto()
    WIDTH = auto()
    HEIGHT = auto()


DIM_CHAR_DICT = {
    DataDim.BATCH: "B",
    DataDim.CHANNEL: "C",
    DataDim.HEIGHT: "H",
    DataDim.WIDTH: "W",
}
""" Enumeration association to char representations.
"""

CHAR_DIM_DICT = dict((v, k) for k, v in DIM_CHAR_DICT.items())
""" Character association to enumeration representations.
"""

class DataFormat:
    BHWC = "BHWC"
    BWHC = "BWHC"
    BCHW = "BCHW"
    BCWH = "BCWH"
    HWC = "HWC"
    WHC = "WHC"
    CHW = "CHW"
    CWH = "CWH"


DATA_FORMAT_INDEX_DIM = {}
""" Storage for pre-defined dimension format dictionaries that map
axis index to dimension type.
"""

DATA_FORMAT_DIM_INDEX = {}
""" Storage for pre-defined dimension format dictionaries that map
dimension type to axis index.
"""


def make_index_dim_dict(data_format: str) -> Dict:
    dim_dict = OrderedDict()
    for i, c in enumerate(data_format):
        dim_dict[i] = CHAR_DIM_DICT[c]
    return dim_dict


def make_dim_index_dict(data_format: str) -> Dict:
    dim_dict = OrderedDict()
    for i, c in enumerate(data_format):
        dim_dict[CHAR_DIM_DICT[c]] = i
    return dim_dict


def add_format(data_format: str):
    global DATA_FORMAT_INDEX_DIM
    DATA_FORMAT_INDEX_DIM[data_format] = make_index_dim_dict(data_format)
    global DATA_FORMAT_DIM_INDEX
    DATA_FORMAT_DIM_INDEX[data_format] = make_dim_index_dict(data_format)


# Create dictionary entries for all formats in DataFormat class
for data_format in list_constants(DataFormat):
    add_format(data_format)
