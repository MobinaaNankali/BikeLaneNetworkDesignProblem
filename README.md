# Bike Lane Network Design

This repository contains the Python code accompanying our paper published in **Transportation Research Part A: Policy and Practice**.

The code selects bike lane locations under a construction budget while accounting for car user equilibrium and bicycle travel times. The solution method combines branch and bound, outer approximation, and column generation.

## Requirements

- Python 3.
- IBM ILOG CPLEX, with its Python API configured in the Python environment used to run the code. The benchmark models exceed the CPLEX Community Edition size limits.
- Python packages:

  ```bash
  python -m pip install docplex llist numpy matplotlib
  ```

## How to run

Run commands from the repository root so the code can find the files in `data/`.

```bash
python main.py
```

The default example is Sioux Falls (`SF_CNDP_30_1`). To run Eastern Massachusetts with 10 candidate links, change the network and instance settings in `main.py` to:

```python
net = 'EasternMassachusetts'
ins = 'EM_CNDP_10_1'
```

Then run `python main.py` again. Ensure these are the active assignments rather than commented examples.

The solver prints the objective value, relative gap, selected links, and elapsed time. Its defaults are `w = 1`, a 40% bike lane travel-time reduction, and a budget equal to 50% of the total candidate construction cost. To change them, set `w` and `bikelane_bonus` on the solver object and pass `budget_percentage` to its constructor. Set `timelimit` on the solver object to control the time limit in seconds.

## Repository contents

- `src/`: network, traffic assignment, and optimization code. The main solver is `NewOA_DNDP_withtimelimit.py`.
- `data/`: network instances and travel demand files.
- `ExperimentRun.py`: batch experiments; configure the instances and run parameters in the script.
- `SensitivityAnalysis_*.py`: sensitivity experiments for budget, objective weight, and bike lane benefits.
- `plot_*.py`: plots of experiment and sensitivity results.
- `tests/`: regression tests.

Please cite the associated paper when using this code in your research.
