#!/usr/bin/env bash

cp gilbreath.cpp CUDASieve/gilbreath.cpp
cp gilbreath.h CUDASieve/gilbreath.h
cp gilbreath.cu CUDASieve/gilbreath.cu
cd CUDASieve
nvcc -ccbin=g++-5 -O3 -I include -L/home/ricson/code/gilbreath/CUDASieve -lcudasieve -std=c++11 -gencode arch=compute_61,code=sm_61 gilbreath.cu gilbreath.cpp -o gilbreath
cd ..
cp CUDASieve/gilbreath gilbreath
