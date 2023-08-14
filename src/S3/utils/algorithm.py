import mahotas
import numpy as np

from S3.s3_logging import get_logger


def watershed(surface, markers, fg):
    # compute watershed
    ws = mahotas.cwatershed(surface, markers)

    # write watershed directly
    get_logger().debug("watershed output: %s %s %f %f",
                       ws.shape, ws.dtype, ws.max(), ws.min())

    # overlay fg and write
    wsFG = ws * fg
    get_logger().debug("watershed (foreground only): %s %s %f %f",
                       wsFG.shape, wsFG.dtype, wsFG.max(),
                       wsFG.min())
    wsFGUI = wsFG.astype(np.uint16)

    return wsFGUI
