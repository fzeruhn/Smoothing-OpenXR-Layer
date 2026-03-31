#!/bin/bash
# Initialize submodules if not already done
if [ -f ".gitmodules" ]; then
  git submodule update --init --recursive
fi