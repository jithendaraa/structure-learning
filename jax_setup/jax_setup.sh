python jaxlibprep.py -C cuda110 -P cp37    # Downloads specified jaxlib version built against specified cuda and python versions.
                      --set-runpath $CUDA_HOME/lib64  # Binary-patches libraries to look in specified additional locations 
                      -t linux  