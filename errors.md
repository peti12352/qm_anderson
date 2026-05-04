(base) v@greengoblin:~/proj/petya/simulations/qm_anderson$ git pull && pixi run sc-dilution
Already up to date.
✨ Pixi task (sc-dilution): python runs/sc_dilution.py: (Measure p_q for SC site-diluted lattice)

=======================================================
ENVIRONMENT
JAX version  : 0.9.2
JAX backend  : gpu
JAX devices  : [CudaDevice(id=0)]
x64 enabled  : True
GPU          : NVIDIA GeForce RTX 5090
Driver       : 580.126.09
VRAM         : 31559 MiB free / 32607 MiB total
Temp / Power : 41°C / 24.41W (limit 575.00W)
=======================================================

Classical p_c(SC) measured: 0.3250 (expected 0.3116)
Traceback (most recent call last):
  File "/home/v/proj/petya/simulations/qm_anderson/runs/sc_dilution.py", line 224, in <module>
    main()
  File "/home/v/proj/petya/simulations/qm_anderson/runs/sc_dilution.py", line 85, in main
    _check_classical_threshold()
  File "/home/v/proj/petya/simulations/qm_anderson/runs/sc_dilution.py", line 72, in _check_classical_threshold
    raise RuntimeError(
RuntimeError: Classical p_c check failed: 0.3250 != 0.3116. Check percolation.py implementation.
(base) v@greengoblin:~/proj/petya/simulations/qm_anderson$ 
