##################################################################
# 2022 - King Abdullah University of Science and Technology (KAUST)
#
# Authors: Nick Luiken, Matteo Ravasi
# Description: Blending operators
##################################################################

import pylops
from pylops.signalprocessing import *


def Blending(nt, nr, ns, dt, overlap, times, nproc=1, dtype='float64'):
    """Blending operator

    Blend seismic shot gather in continuos mode based on pre-defined sequence of firing times.

    Parameters
    ----------
    nt : :obj:`int`
        Number of time samples
    nr : :obj:`int`
        Number of receivers
    ns : :obj:`int`
        Number of sources
    dt : :obj:`float`
        Time sampling in seconds
    overlap : :obj:`float`
        Percentage of overlap between consecutive shots
    times : :obj:`np.ndarray`
        Dithering ignition times (to be added to periodic nominal firing times
    nproc : :obj:`int`, optional
        Number of processors used when applying operator
    dtype : :obj:`str`, optional
        Operator dtype

    Returns
    -------
    Dop : :obj:`pylops.LinearOperator`
        Blending operator

    """
    # Amount of padding needed
    pad = int(overlap * nt)

    # Create operator
    OpShiftPad = []
    for i in range(ns):
        # Create padding and shifting for each shot
        PadOp = pylops.Pad((nt, nr), ((pad * i, pad*(ns - 1 - i )), (0, 0)), dtype=dtype)
        ShiftOp = pylops.signalprocessing.Shift((pad * (ns - 1) + nt, nr), times[i], dir=0,
                                                sampling=dt, real=False, dtype=dtype)
        Operator = ShiftOp * PadOp
        OpShiftPad.append(Operator)
    Top = pylops.Transpose(dims=(ns, nr, nt), axes=(0, 2, 1))
    # Combine shift operators from all shots
    Top = pylops.Transpose(dims=(ns, nr, nt), axes=(0, 2, 1))
    return pylops.HStack(OpShiftPad, nproc=nproc) * Top