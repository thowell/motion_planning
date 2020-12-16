# Direct Motion Planning

## classic examples
- [X] double integrator
- [X] acrobot
- [ ] robotic arm
- [ ] quadrotor

## contact-implicit trajectory optimization examples
- update models: impulse -> force
- [X] particle
- [X] simple manipulation
- [X] cyberdrift
- [X] cyberjump
- [X] box drop
- [X] box on corner
- [X] raibert hopper (2D)
- [X] raibert hopper (2D) vertical gait
- [X] raibert hopper (2D) flip
- [ ] simple walker (2D)
- [X] hopper (3D)
- [ ] hopper (3D) wall scaling
- [X] miniature golf
- [ ] ball-in-cup robot arm
- [ ] ball-in-cup quadrotor
- [X] biped (2D)
- [X] quadruped (2D)
- [X] quadruped (2D) over box
- [X] quadruped (2D) backflip
- [ ] quadruped (3D)
- [ ] ant (3D)
- [ ] snake (3D)
- [ ] atlas

## direct policy optimization examples
We provide the [examples](src/examples/direct_policy_optimization) from [Direct Policy Optimization using Deterministic Sampling and Collocation](https://arxiv.org/abs/2010.08506). Optimizing the policies requires [SNOPT](https://en.wikipedia.org/wiki/SNOPT) and resources for its installation are available [here](src/solvers/snopt.jl). The trajectories and policies from these examples have been saved and can be loaded in order to run the policy simulations and visualizations.

- update: discrete disturbance dynamics, plots

LQR
- [X] double integrator
- [X] planar quadrotor

motion planning
- [X] pendulum
- [X] autonomous car
- [X] cart-pole
- [X] rocket
- [X] quadrotor
- [X] biped

## installation
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```julia
pkg> add https://github.com/thowell/DirectMotionPlanning.jl
```

## TODO
- [X] direct policy optimization implementation
	- [ ] update paper visualizations
	- [X] save TO and DPO trajectories
	- [ ] solve DPO to tighter tolerances
- [ ] check for SNOPT installation
- [ ] parallelize objective + constraint evaluations
- [ ] tests
- [ ] visualization dependencies
	- [ ] select default background
	- [ ] set default views
- [X] nonlinear objective (stage wise)
- [X] constraints (stage wise)
- [X] discrete dynamics parameterization
- [ ] embed animations in README
- [ ] dispatch over model type for free final time
- [ ] analytical velocity objective gradient
- [ ] large-scale augmented Lagrangian solver
- [ ] contact simulator
