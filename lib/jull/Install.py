#!/usr/bin/env python

"""
Install.py tool to download, unpack, build, and link to the ALL library
used to automate the steps described in the README file in this dir
"""

from __future__ import print_function
import sys, os, subprocess, shutil, tarfile
from argparse import ArgumentParser

sys.path.append('..')
from install_helpers import fullpath, geturl, get_cpus, checkmd5sum, getfallback

parser = ArgumentParser(prog='Install.py', description="LAMMPS library build wrapper script")

# settings

version = "0.9.3"

# known md5 checksums for different ALL versions. used to validate the download.
checksums = {
        '0.9.2' : '2fcc8bcb60f33fa0369e8f44a5c4b884',
        '0.9.3' : '9fc008711a7dfaf35e957411f8ed1504'
        }

# extra help message

HELP = """
Syntax from src dir: make lib-jull args="-b"
                 or: make lib-jull args="-p /path/to/jull"
Syntax from lib dir: python Install.py -b
                 or: python Install.py -p /path/to/jull

Example:

make lib-jull args="-b"   # download/build in lib/jull/ALL
make lib-jull args="-p $HOME/ALL" # use existing ALL installation in $HOME
"""

# parse and process arguments

pgroup = parser.add_mutually_exclusive_group()
pgroup.add_argument("-b", "--build", action="store_true",
                    help="download and build the ALL library")
pgroup.add_argument("-p", "--path",
                    help="specify folder of existing ALL installation")
parser.add_argument("-v", "--version", default=version,
                    help="set version of ALL to download and build (default: %s)" % version)

args = parser.parse_args()

# print help message and exit, if neither build nor path options are given
if not args.build and not args.path:
  parser.print_help()
  sys.exit(HELP)

buildflag = args.build
pathflag = args.path is not None
version = args.version
url = "https://gitlab.jsc.fz-juelich.de/SLMS/loadbalancing/-/archive/v%s/loadbalancing-v%s.tar.gz" % (version, version)


homepath = fullpath(".")
ALL_path = os.path.join(homepath, "loadbalancing-v%s" % version)

if pathflag:
  ALL_path = args.path
  if not os.path.isdir(os.path.join(ALL_path, "include")):
    sys.exit("ALL include path for %s does not exist" % ALL_path)
  if (not os.path.isdir(os.path.join(ALL_path, "lib64"))) \
     and (not os.path.isdir(os.path.join(ALL_path, "lib"))):
    sys.exit("ALL lib path for %s does not exist" % ALL_path)
  ALL_path = fullpath(ALL_path)

# download and unpack ALL tarball

if buildflag:
  print("Downloading ALL ...")
  filename = "%s/loadbalancing-v%s.tar.gz" % (homepath, version)
  fallback = getfallback('loadbalancing', url)
  try:
    geturl(url, filename)
  except:
    geturl(fallback, filename)

  # verify downloaded archive integrity via md5 checksum, if known.
  if version in checksums:
    if not checkmd5sum(checksums[version], filename):
      print("Checksum did not match. Trying fallback URL", fallback)
      geturl(fallback, filename)
      if not checkmd5sum(checksums[version], filename):
        sys.exit("Checksum for ALL library does not match for fallback, too.")

  print("Unpacking ALL tarball ...")
  if os.path.exists(ALL_path):
    shutil.rmtree(ALL_path)
  tarname = os.path.join(homepath, "%s.tar.gz" % ALL_path)
  if tarfile.is_tarfile(tarname):
    tgz = tarfile.open(tarname)
    tgz.extractall(path=homepath)
    os.remove(tarname)
  else:
    sys.exit("File %s is not a supported archive" % tarname)

  # build ALL
  print("Building ALL ...")
  n_cpu = get_cpus()
  build_dir = os.path.join(homepath, 'build')
  cmd = 'gotoCleanDir() { test -d $1 && rm -r $1 ; mkdir -p $1 ; cd $1 ; } ; gotoCleanDir "%s" ; cmake "%s" ; cmake --build ./ --parallel "%d" && cmake --install ./ --prefix ./' % (build_dir, ALL_path, n_cpu)
  try:
    txt = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    print(txt.decode('UTF-8'))
  except subprocess.CalledProcessError as e:
    sys.exit("CMake failed with:\n %s" % e.output.decode('UTF-8'))

# create 2 links in lib/jull to ALL include/lib dirs

print("Creating links to ALL include and lib files")
if os.path.isfile("include") or os.path.islink("include"):
  os.remove("include")
if os.path.isfile("lib") or os.path.islink("lib"):
  os.remove("lib")
if buildflag:
  os.symlink(os.path.join(homepath, 'build', 'include'), 'include')
  os.symlink(os.path.join(homepath, 'build', 'lib'), 'lib')
else:
  os.symlink(os.path.join(ALL_path, 'include'), 'include')
  if os.path.isdir(os.path.join(ALL_path, "lib64")):
    os.symlink(os.path.join(ALL_path, 'lib64'), 'lib')
  else:
    os.symlink(os.path.join(ALL_path, 'lib'), 'lib')
