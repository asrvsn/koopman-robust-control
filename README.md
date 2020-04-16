
# Generating uncertainty sets of transfer operators

This project contains the experimental framework for generating perturbations of nonlinear dynamical systems via a nominal Koopman operator, per "An MCMC Method for Uncertainty Set Generation via Operator-Theoretic Metrics," by A. Srinivasan and N. Takeishi submitted to IEEE CDC. 


Below is the library organization. To run any module, simply use `python -m FOLDER.MODULE`.
```bash
├── sampler
│   ├── ugen.py 			# High-level MCMC procedure for uncertainty set generation
│   ├── hmc.py 				# PyTorch autograd-based Hamiltonian Monte Carlo for tensor-valued arguments with support for constraint-based reflection
│   ├── hmc_nuts.py 		# No U-Turn Sampler integrator for HMC (not used in experiments)
│   ├── hmc_parallel.py 	# Parallel HMC sampler from a specified prior over initial conditions 
│   ├── kernel.py 			# Positive-definite kernel over dynamical systems (autograd-compliant implementation of Ishikawa et al., https://arxiv.org/abs/1805.12324)
│   ├── reflections.py 		# Various boundary conditions for HMC 
│   ├── features.py 		# Observables & kernels for Koopman operator
│   ├── operators.py 		# Dynamic Mode Decomposition & variants
│   └── utils.py 	
├── experiments				# Examples of uncertainty set generation for prediction & control (see below)
└── ...
```

Please see the below sections for usage of the above procedures and replication of numerical examples, from prediction to control. 


## Perturbations of LTI systems
![](https://github.com/ooblahman/koopman-robust-control/blob/master/figures/2x2_comparison_pdf.png)

1. Configure candidate systems in `systems/lti2x2.py`
2. Run `python -m experiments.2x2_perturb` with `method = 'discounted_kernel'` to generate an uncertainty set
3. Run `python -m experiments.2x2_plot` with the correct filename to produce trace-determinant plots 

## Perturbations of nonlinear systems
### Duffing Oscillator
![](https://github.com/ooblahman/koopman-robust-control/blob/master/figures/duffing_kernel.png)

1. Configure Duffing equation parameters in `experiments/duffing_perturb.py` (unforced only for poly obs.)
2. Run `python -m experiments.duffing_perturb` with `method = kernel'` to generate perturbations
3. Run `python -m experiments.duffing_plot` to generate attractor basin visualizations

### Van der Pol Oscillator
![](https://github.com/ooblahman/koopman-robust-control/blob/master/figures/vdp_small_step.png)

1. Configure VDP parameters in `experiments/vdp_perturb.py`
2. Run `python -m experiments.vdp_perturb` with `method = kernel` or `method = constrained_kernel'` to generate uncertainty set
3. Run `python -m experiments.vdp_plot` to show phase-space results.


## Perturbing a custom system (instructions)

1. Compute a nominal Koopman operator for the system. (See `experiments/duffing_perturb.py` for an example using the `Observable` class from `sampler/features.py`. Any algorithm can be used here.)
2. Call `sampler.ugen.perturb(...)` with `model = my_koopman_op, method = 'kernel'`, and any other arguments specified in the file. 
3. Adjust parameters of parallel HMC (`hmc_step`, `hmc_leapfrog`, `n_ics`, `ic_step`, `ic_leapfrog`) until the desired stationary distribution is reached (i.e. MCMC is adequately mixed; `sampler.ugen.perturb()` will return the posterior distribution as its second result, which can be used for visual/numerical verification.) 
4. Use the resulting uncertainty set for robust prediction & control.

## (In progress) Robust control & scenario optimization examples

We're still working on tuning the method to achieve noteworthy improvements for model-predictive controllers facing heavy observation noise, check back in later! Some preliminary results...

![](https://github.com/ooblahman/koopman-robust-control/blob/master/figures/rc_prelim.png)

