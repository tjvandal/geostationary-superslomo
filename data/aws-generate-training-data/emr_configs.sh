# AWS EMR 5.20.0 bootstrap script for installing/configuring torch and xarray

# ----------------------------------------------------------------------
#                    Install Environment              
# ----------------------------------------------------------------------
aws s3 cp s3://nex-goes-slowmo/emr-goes-dependencies.tar.gz .
mkdir $HOME/src
tar -xvf emr-goes-dependencies.tar.gz -C $HOME/src/

echo "export PYTHONPATH=$HOME/src/:$PYTHONPATH" >> $HOME/.bashrc
echo "AWS_ACCESS_KEY_ID=AKIAI3SPU3NJ4U733ONQ" >> $HOME/.bashrc
echo "AWS_SECRET_ACCESS_KEY=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" >> $HOME/.bashrc

#sudo yum install python-setuptools
#sudo easy_install pip
sudo pip install -U --no-cache-dir boto boto3 xarray netcdf4 scipy pillow torch torchvision opencv-python awscli
sudo yum install libXext libSM libXrender # opencv dependencies
source $HOME/.bashrc
