import logging
import os

from utils import makedirs

if (not os.path.exists(r'.\different_A_initial_interpolation_extrapolation_longtime')):
    makedirs(r'.\different_A_initial_interpolation_extrapolation_longtime')

log_filename = r'different_A_initial_interpolation_extrapolation_longtime/different_A_initial_biochemical.txt'
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.info("biochenmical")

logging.shutdown()

log_filename = r'different_A_initial_interpolation_extrapolation_longtime/different_A_initial_heat.txt'
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.info("heat")
