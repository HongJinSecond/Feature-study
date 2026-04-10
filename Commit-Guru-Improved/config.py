"""
file: config.py
description: Reads the config.json info into a varible
"""
import json
#from StringIO import StringIO
config = json.load(open('./config.json'))