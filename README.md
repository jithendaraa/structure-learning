# structure-learning

## Setup

1. Create your virtual environment: `virtualenv --no-download ~/causal`
2. Add the following to your bashrc file if you are on computecanada:
   ```
   alias act_causal='source ~/causal/bin/activate' 
   module load python/3.7   
   module load cuda/11.0    
   export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_PATH}   
   act_causal   
   ``` 
   and run `source /path/to/bashrc/file`

3. If you're not on computecanada, make sure you install `jaxlib==0.1.69`. If you <i>are</i> on CC, then install `jaxlib` by running the following: 
      ```
      cd jax_setup/ 
      chmod +x jax_setup.sh 
      ./jax_setup.sh 
      cd tmp/repacked/ 
      pip install jaxlib-0.1.69+cuda110-cp37-none-linux_x86_64.whl 
      cd ../../..
      ```
4. Install all other dependencies with `pip install -r requirements.txt`

## Running jobs

1. `jobs.sh` is responsible for running jobs. It works by calling a shell script under `script_runners` (depending on which model - specified as an arg while running `jobs.sh`) which in turn calls `scripts/run_job.sh` which submits an `SBATCH` job.
2. For the first run: Make `jobs.sh` executable by `chmod +x jobs.sh`
3. Run `./jobs.sh x y z t` where `x` is a number between 0-6 (see supported models), `y` is the dataset ('clevr' or 'er'), `z` currently supports only 'train', and `t` is time as you would specify for submitting a SLURM job.
4. PS: please change line 8 of `scripts/run_job.sh` `SBATCH --mail=...` to your mail address to get notified about your job status.


## Supported models
1. Slot Attention
2. VCN
3. Image VCN
4. Slot Image VCN
5. DIBS
6. VAE-DIBS (still needs fixes; not reliable)
7. Decoder-DIBS

6 and 7 are to study causal inference after projecting data to higher dimensions, either linearly or non-linearly.



## Bibliography
Parts of the code were taken from <a href='https://github.com/google-research/google-research/tree/master/slot_attention'>Slot attention</a>, <a href="https://github.com/yannadani/vcn_pytorch">VCN</a>, and <a href="https://github.com/larslorch/dibs">DiBS</a>:


```
@article{locatello2020object,
      title={Object-Centric Learning with Slot Attention},
      author={Locatello, Francesco and Weissenborn, Dirk and Unterthiner, Thomas and Mahendran, Aravindh and Heigold, Georg and Uszkoreit, Jakob and Dosovitskiy, Alexey and Kipf, Thomas},
      journal={arXiv preprint arXiv:2006.15055},
      year={2020}
}

@article{annadani2021variational,
      title={Variational Causal Networks: Approximate Bayesian Inference over Causal Structures},
      author={Annadani, Yashas and Rothfuss, Jonas and Lacoste, Alexandre and Scherrer, Nino and Goyal, Anirudh and Bengio, Yoshua and Bauer, Stefan},
      journal={arXiv preprint arXiv:2106.07635},
      year={2021}
}

@article{lorch2021dibs,
      title={DiBS: Differentiable Bayesian Structure Learning},
      author={Lorch, Lars and Rothfuss, Jonas and Sch{\"o}lkopf, Bernhard and Krause, Andreas},
      journal={arXiv preprint arXiv:2105.11839},
      year={2021}
}
```

