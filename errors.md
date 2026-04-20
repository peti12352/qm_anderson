(base) v@greengoblin:~/proj/petya/simulations/qm_anderson$ pixi run python runs/sc_benchmark.py --refit --w-min 16.0 --w-max 17.5
Refit mode: loading data/sc_benchmark.h5, FSS window [16.0, 17.5]
/home/v/proj/petya/simulations/qm_anderson/src/fss.py:148: UserWarning: χ²/dof = 7.62 >> 1. FSS fit is poor. Consider: (1) wider x range, (2) more system sizes, (3) higher poly_order, (4) corrections to scaling.
  warnings.warn(

# fit range: (16.0, 17.5)
W_c = 16.3502 ± 0.0059 (expected 16.53)
ν = 1.4775 ± 0.0175 (expected 1.571)
χ²/dof = 7.620
(base) v@greengoblin:~/proj/petya/simulations/qm_anderson$ 
