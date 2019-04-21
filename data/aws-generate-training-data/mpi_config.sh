# AWS EMR 5.20.0 bootstrap script for installing/configuring torch and xarray

# ----------------------------------------------------------------------
#                    Install Environment              
# ----------------------------------------------------------------------
echo "export CC=/usr/lib64/openmpi/bin/mpicc" >> $HOME/.bashrc
echo "export PATH=/usr/lib64/openmpi/bin:$PATH" >> $HOME/.bashrc
echo "export PYTHONPATH=$HOME/src/:$PYTHONPATH" >> $HOME/.bashrc
echo "AWS_ACCESS_KEY_ID=AKIAI3SPU3NJ4U733ONQ" >> $HOME/.bashrc
echo "AWS_SECRET_ACCESS_KEY=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" >> $HOME/.bashrc

pip install mpi4py torch torchvision opencv-python scipy xarray --user
source $HOME/.bashrc
