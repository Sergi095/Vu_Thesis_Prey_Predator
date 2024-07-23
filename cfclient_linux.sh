#!/bin/bash


# Install required packages
sudo apt-get update
sudo apt-get install -y git python3-pip libxcb-xinerama0 libxcb-cursor0
pip3 install --upgrade pip

# Add the user to the plugdev group
sudo groupadd plugdev
sudo usermod -a -G plugdev $USER

# Create udev rules
echo '# Crazyradio (normal operation)' | sudo tee /etc/udev/rules.d/99-bitcraze.rules > /dev/null
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"' | sudo tee -a /etc/udev/rules.d/99-bitcraze.rules > /dev/null
echo '# Bootloader' | sudo tee -a /etc/udev/rules.d/99-bitcraze.rules > /dev/null
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="0101", MODE="0664", GROUP="plugdev"' | sudo tee -a /etc/udev/rules.d/99-bitcraze.rules > /dev/null
echo '# Crazyflie (over USB)' | sudo tee -a /etc/udev/rules.d/99-bitcraze.rules > /dev/null
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0664", GROUP="plugdev"' | sudo tee -a /etc/udev/rules.d/99-bitcraze.rules > /dev/null

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Install cfclient
pip install cfclient

