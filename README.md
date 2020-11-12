# Direct Motion Planning

## classic examples
- [X] double integrator
- [X] acrobot
- [ ] robotic arm
- [X] biped

## contact-implicit trajectory optimization examples
- [X] particle
- [X] simple manipulation
- [X] cyberdrift
- [X] cyberjump
- [X] box drop
- [X] box on corner
- [X] raibert hopper
- [X] raibert hopper vertical gait
- [X] raibert hopper flip
- [X] hopper (3D)
- [ ] hopper (3D) wall scaling
- [X] miniature golf
- [ ] ball-in-cup robot arm
- [ ] ball-in-cup quadrotor
- [ ] biped
- [ ] quadruped
- [ ] atlas

## direct policy optimization examples
We provide the [examples](src/examples/direct_policy_optimization) from [Direct Policy Optimization using Deterministic Sampling and Collocation](https://arxiv.org/abs/2010.08506). Optimizing the trajectories requires [SNOPT](https://en.wikipedia.org/wiki/SNOPT) and resources for its installation are available [here](src/solvers/snopt.jl). These trajectories have been saved and can be loaded in order to run the policy simulations and visualizations.

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

## TODO
- [X] direct policy optimization implementation
	- [ ] paper visualizations
	- [X] save TO and DPO trajectories
- [ ] check for SNOPT installation
- [ ] parallelize objective + constraint evalutations
- [ ] tests
- [ ] visualization dependencies
	- [ ] select default background
- [X] nonlinear objective (stage wise)
- [X] constraints (stage wise)
- [ ] embed animations in README
