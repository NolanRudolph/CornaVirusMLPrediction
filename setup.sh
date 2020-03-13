#!/bin/bash
# Installs required prerequisites for running corona_model.py

# Confirms if user wants to install new software
echo "This program installs prerequsites for this repository: Python3, pip3, \
python3-lib-pandas, python3-lib-matplotlib, and python3-lib-sklearn \
will be installed if these files do not exist. This is preceeded by a \
system update. All of this will not be harmful to your computer apart \
from storage issues.";
echo -n "Do you wish to proceed? [y / n] ";
read userInput;
if [[ $userInput =~ [Nn] ]]
then
	echo "Exiting...";
	exit(1);
elif ![[ $userInput =~ [Yy] ]]
then
	echo "Please respond with y or n. Exiting...";
	exit(1);
fi

# Update system to not ruin user's computer
echo "Updating...";
sudo apt update > /dev/null;

# Install python3 if not already installed
if [ -z $(which python3) ]
then
	echo "Installing python3.7...";
	sudo apt-get install -y python3 > /dev/null;
fi

# Install pip3 if not already installed
if [ -z $(which pip3) ]
then
	echo "Installing pip3...";
	sudo apt-get install -y pip3 > /dev/null;
fi

# Install pandas if not already installed
if [ -z $(pip list | grep pandas) ]
then
	echo "Installing Panda...";
	sudo pip3 install pandas > /dev/null;
fi

# Install matplotlib if not already installed
if [ -z $(pip list | grep matplotlib) ]
then
	echo "Installing Matplotlib...";
	pip3 install matplotlib > /dev/null;
fi

echo "Finished Setup!"
