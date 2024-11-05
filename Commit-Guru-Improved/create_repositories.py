
from orm.repository import Repository
import argparse
import db

parser = argparse.ArgumentParser(description='Add repository.')
parser.add_argument('name', help='repository name')
parser.add_argument('url', help='repository url')

args=parser.parse_args()
print(args)

session = db.Session()
session.merge(Repository({'name':args.name, 'url':args.url, 'status':'Waiting to be Ingested'}))        
session.commit()
session.close()
