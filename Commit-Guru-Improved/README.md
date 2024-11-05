[![DOI](https://zenodo.org/badge/192493650.svg)](https://zenodo.org/badge/latestdoi/192493650)


# CommitGuru - Improved
==========================================
Update by Zuowei Chen
2024/11/5
2021113561@stu.hit.edu.cn

Ingests and analyzes code repositories.

==> This implementation is a modification of the original Commit.Guru code to enable processing git repositories with commit messages in Chinese and to collect information about the timestamp when the defects linked to software changes were fixed.

## Installation
1. Clone this repository in to an empty directory
2. Copy the `./config.example.json` to `./config.json` and change the
the configurations. All fields are required.

Db: information relating to your postgresql database setup
logging: information about how to write logging information
gmail: gmail account to be used to send cas notifications
repoUpdates: how often repositories should be updated for new commits
system: how many worker threads the cas system can use to analyze and ingest repos.

### Dependencies
make share you are running in **Linux**
Additional Instructions are available in SETUP.md
* Python  >= 3.3 and <= 3.6
* Pip for Python Version > 3.3 and < 3.6
* Git > 1.7
* R
* python-dev
* rpy2
* requests
* dateutil
* sqlalchemy
* py-postgresql
* GNU grep
* MonthDelta

<!-- ###Setting up python3.3 virtual env on Ubuntu
* Assumes you are working on Ubuntu 12.04

Install python3.3 using the deadsnakes PPA:

```
sudo apt-get install python-software-properties
sudo add-apt-repository ppa:fkrull/deadsnakes
sudo apt-get update
sudo apt-get install python3.3
``` -->
<!-- 
Version 1.7.1.2 of virtual env that comes with Ubuntu 12.04 is not compatibale with python3.3.
Therefore, we must installa new version so that we can setup a working virutal environment. First,
you must uninstall the current python-virtualenv:

```
sudo apt-get remove python-virtualenv
```

Next, install the latest easy_install:

```
wget http://peak.telecommunity.com/dist/ez_setup.py
sudo python ez_setup.py
```

Next, install pip and the virtualenv:

```
sudo easy_install pip
sudo pip install virtualenv
virtualenv --no-site-packages --distribute -p /usr/bin/python3.3 ~/.virtualenvs/pywork3
```

By default, typically we don't have the python-dev available for python3 on Ubuntu after setting up a new
virtual environment for it and so have to install it as it's a dependency for rpy2. Install this with apt-get:

```
sudo apt-get install python3.3-dev
```

Now, we are finally ready to set up our virtual environment:

```
virtualenv -p /usr/bin/python3.3 /path/to/new/virtual/environment
```

To activate the virtual env:

```
source /path/to/new/virtual/environemnt/bin/activate
```

Type `deactiviate` to exit the virtual env -->

### Installing rpy2
<!-- * Assumes you are working on Ubuntu 12.04 and python 3.3

Getting rpy2 to work can be a bit tricky. First, make sure R is installed. To do this, first
get the repository SSL key and import it to apt by doing

  ```
  gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
  gpg -a --export E084DAB9 | sudo apt-key add -
  ``` -->

<!-- Then, Edit the list of sources `gksudo gedit /etc/apt/sources.list` and add the following repo at the bottom:`deb http://cran.ma.imperial.ac.uk/bin/linux/ubuntu precise/` -->

<!-- Finally, we can install R by running the following commands:

  ```
  sudo apt-get update
  sudo apt-get install r-base
  ``` -->

<!-- Now we are ready to install rpy2. Make sure python version 3 or greater is in use (3.2 is not compatibale, however), such as by using a virtualenv and run -->

```
pip install rpy2
```

### Additional Pip Packages
Install the following packages by doing `pip install `  and then the package
name. Make sure you are using python3, such as using a virtualenv if using Ubuntu.

* SQL Alchemy (sqlalchemy)
<!-- * Py-PostgreSQL (py-postgresql) (no use anymore) -->
* psycopg2
* requests (requests)
* python-dateutil (python-dateutil)
* any package the runtime pip prompts you that are missing.

To install the MonthDelta package, simply do: `pip install https://pypi.python.org/packages/source/M/MonthDelta/MonthDelta-1.0b.tar.bz2`

### Attetion
make sure that this project need the dataset tool PostgreSQL

### First-Time Database Setup
Set up the database for the first time by running `python script.py initDb`

## Usage
In a terminal, type `nohup python script.py & ' to start the code repo analyzer and run it in the background.

## OFFLine
Update by Zuowei Chen 2024-10-28
I have found if we once use the script.py to extract datasets, everytime it will use **git clone** to clone the project. However, the clone process may be interrupted due to the network problem and we need restart this tool again, which is very hassle and waste of time, so I have integrates a offline version of this tool, the usage can be seen below:
```
python script.py initDb
python create_repositories.py {project_name} {url} #In offline version, {url} can be any string since we do not use it.
python offline.py {project_name} #make sure that we have already clone the project to ./CASRepos/git/
```