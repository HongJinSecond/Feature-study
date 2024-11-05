"""
file: db.py
author: Ben Grawi <bjg1568@rit.edu>
date: October 2013
description: Holds the db connection info
"""
from config import *
import sqlalchemy
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Session = sessionmaker()

connection_args = []
connection_args.append(config['db']['type'])

if config['db']['adapter']:
	connection_args.extend(['+',config['db']['adapter']])

connection_args.append('://')

if config['db']['username']:
	connection_args.append(config['db']['username'])
	if config['db']['password']:
		connection_args.extend([':',config['db']['password']])
	connection_args.append('@')

if config['db']['host']:
	connection_args.append(config['db']['host'])
	if config['db']['port']:
		connection_args.extend([':',config['db']['port']])
connection_args.extend(['/',config['db']['database']])

engine = sqlalchemy.create_engine(''.join(connection_args)) # the value of pool_size has to be less than the max_connections to postgres.
Session.configure(bind=engine)
Base = declarative_base()
