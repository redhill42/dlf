#!/usr/bin/env bash
set -e

# This script builds various binary artifacts from a checkout of the DLF
# source code.
#
# Requirements:
# - The current directory should be a checkout of the DLF source code.
#   Whatever version is checked out will be built.
# - The VERSION file, at the root of the repository, should exist, and
#   will be used as DLF binary version and package version.
# - The hash of the git commit will also be included in the DLF binary,
#   with the suffix -unsupported if the repository isn't clean.
# - The script is intended to be run inside the docker container specified
#   in the Dockerfile at the root of the source. In other words:
#   DO NOT CALL THIS SCRIPT DIRECTLY.
# - The right way to call this script is to invoke "make" from your checkout
#   of the DLF repository.
#   The makefile will do a "docker build -t dlf-dev . and then
#   "docker run build/make.sh" in the resulting image.

set -o pipefail

if command -v git &> /dev/null && [ -d .git ] && git rev-parse &> /dev/null; then
  GITCOMMIT=$(git rev-parse --short HEAD)
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    GITCOMMIT="$GITCOMMIT-unsupported"
    echo "#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "# GITCOMMIT = $GITCOMMIT"
    echo "# The version you are building is listed as unsuuported because"
    echo "# there are some files in the git repository that are in an uncommitted state."
    echo "# Commit these changes, or add to .gitignore to remove the -unsupported from the version."
    echo "# Here is the current list:"
    git status --porcelain --untracked-files=no
    echo "#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  fi
elif [ "$DLF_GITCOMMIT" ]; then
  GITCOMMIT="$DLF_GITCOMMIT"
else
  echo >&2 "#error: .git directory missing and DLF_GITCOMMIT not specified"
  echo >&2 "  Please either build with the .git directory accessible, or specify the"
  echo >&2 "  exact (--short) commit hash you are building using DLF_GITCOMMIT"
  echo >&2 "  future accountability in diagnosing build issues.  Thanks!"
  exit 1
fi

# Compute number physical CPU cores
let "cores = $(lscpu | awk '/^CPU\(s\)/{print $2}') / $(lscpu | awk '/^Thread/{print $4}')"

cd bundles
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -- -j $cores
