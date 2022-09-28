conda install numpy
cd jax_setup/
python jaxlibprep.py -V 0.1.69 -C cuda11 -P cp37 --set-runpath $CUDA_HOME/lib64 -t linux
cd tmp/repacked/
pip install jaxlib-0.1.69+cuda110-cp37-none-linux_x86_64.whl
cd ../..
pip install jax==0.2.18 ott-jax==0.1.14 cdt==0.5.23
conda install -c anaconda scikit-learn
pip install matplotlib
pip install optax==0.0.9
pip install dm-haiku==0.0.4
conda install -c conda-forge tensorflow-probability
pip install torch==1.10.0
conda install -c conda-forge wandb
pip install graphical-models
conda install -c conda-forge ipdb python-igraph
