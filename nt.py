import cv2
import numpy as np
import logging

from networktables import NetworkTables

# start vision networkTables
#
# argments:
#   *server = IP address of table.
#
# return
def initTable(server="10.3.34.22"):
    # necessary to see networkTables
    logging.basicConfig(level=logging.DEBUG)

    NetworkTables.initialize(server)
    VISION_TABLE = NetworkTables.getTable('vision')

    return VISION_TABLE

# send values to networkTables
#
# arguments:
#   *NETWORK_TABLE = networkTable to send to
#   *data = map of values to send
def sendData(NETWORK_TABLE, data):
    for key, value in data.iteritems():
        if type(value) is int or type(value) is np.float64:
            NETWORK_TABLE.putNumber(key, value)
        elif type(value) is str:
            NETWORK_TABLE.putString(key, value)
        elif type(value) is bool:
            NETWORK_TABLE.putBoolean(key, value)

