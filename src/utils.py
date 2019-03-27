#!/usr/bin/env python
# encoding: utf-8

import yaml
import logging

LOG_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'


def import_config(conf_name):
    """
    Load YAML configuration file
    """
    with open("../etc/config.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    cfg_sect = cfg[conf_name]
    return cfg_sect


def set_log_config(logfile, level):
    """
    Configure log file
    """
    if logfile:
        logging.basicConfig(filename=logfile,
                            format=LOG_FORMAT,
                            level=level)
    else:
        logging.basicConfig(format=LOG_FORMAT,
                            level=level)
