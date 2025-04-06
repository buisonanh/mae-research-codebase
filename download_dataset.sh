#!/bin/bash

# If param is --name is rafdb
if [ "$1" = "--name" ] && [ "$2" = "rafdb" ]; then
    echo "Downloading RAF-DB dataset..."
    mkdir -p datasets
    curl -L -o datasets/raf-db-dataset.zip https://www.kaggle.com/api/v1/datasets/download/shuvoalok/raf-db-dataset
    # Create the data directory if it doesn't exist
    mkdir -p datasets/raf-db-dataset
    unzip datasets/raf-db-dataset.zip -d datasets/raf-db-dataset
    rm datasets/raf-db-dataset.zip
fi

# If param is --name is affectnet
if [ "$1" = "--name" ] && [ "$2" = "affectnet" ]; then
    echo "Downloading AffectNet dataset..."
    mkdir -p datasets
    curl -L -o datasets/AffectNet.zip https://www.kaggle.com/api/v1/datasets/download/thienkhonghoc/affectnet
    # Create the data directory if it doesn't exist
    mkdir -p datasets/affectnet
    unzip datasets/AffectNet.zip -d datasets/affectnet
    rm datasets/AffectNet.zip
fi

# If param is --name is keypoints
if [ "$1" = "--name" ] && [ "$2" = "keypoints" ]; then
    echo "Downloading Keypoints dataset..."
    mkdir -p datasets
    curl -L -o datasets/keypoints.zip https://www.kaggle.com/api/v1/datasets/download/bravo03/facial-detection-keypoints
    # Create the data directory if it doesn't exist
    mkdir -p datasets/keypoints
    unzip datasets/keypoints.zip -d datasets/keypoints
    rm datasets/keypoints.zip
fi
