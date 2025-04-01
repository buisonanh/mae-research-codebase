#!/bin/bash

# If param is --name is rafdb
if [ "$1" = "--name" ] && [ "$2" = "rafdb" ]; then
    echo "Downloading RAF-DB dataset..."
    curl -L -o /home/sonanhbui/projects/mae-research-codebase/datasets/raf-db-dataset.zip https://www.kaggle.com/api/v1/datasets/download/shuvoalok/raf-db-dataset
    # Create the data directory if it doesn't exist
    mkdir -p /home/sonanhbui/projects/mae-research-codebase/datasets/raf-db-dataset
    unzip /home/sonanhbui/projects/mae-research-codebase/datasets/raf-db-dataset.zip -d /home/sonanhbui/projects/mae-research-codebase/datasets/raf-db-dataset
    rm /home/sonanhbui/projects/mae-research-codebase/datasets/raf-db-dataset.zip
fi

# If param is --name is affectnet
if [ "$1" = "--name" ] && [ "$2" = "affectnet" ]; then
    echo "Downloading AffectNet dataset..."
    curl -L -o /home/sonanhbui/projects/mae-research-codebase/datasets/AffectNet.zip https://www.kaggle.com/api/v1/datasets/download/thienkhonghoc/affectnet
    # Create the data directory if it doesn't exist
    mkdir -p /home/sonanhbui/projects/mae-research-codebase/datasets/affectnet
    unzip /home/sonanhbui/projects/mae-research-codebase/datasets/AffectNet.zip -d /home/sonanhbui/projects/mae-research-codebase/datasets/affectnet
    rm /home/sonanhbui/projects/mae-research-codebase/datasets/AffectNet.zip
fi

# If param is --name is keypoints
if [ "$1" = "--name" ] && [ "$2" = "keypoints" ]; then
    echo "Downloading Keypoints dataset..."
    curl -L -o /home/sonanhbui/projects/mae-research-codebase/datasets/keypoints/keypoints.zip https://www.kaggle.com/api/v1/datasets/download/bravo03/facial-detection-keypoints
    # Create the data directory if it doesn't exist
    mkdir -p /home/sonanhbui/projects/mae-research-codebase/datasets/keypoints
    unzip /home/sonanhbui/projects/mae-research-codebase/datasets/keypoints/keypoints.zip -d /home/sonanhbui/projects/mae-research-codebase/datasets/keypoints
    rm /home/sonanhbui/projects/mae-research-codebase/datasets/keypoints/keypoints.zip
fi
