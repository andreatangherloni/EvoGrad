# EvoGrad: Fully Differentiable Evolutionary and Swarm-Based Optimisation Algorithms

## EvoGrad

**EvoGrad** is a framework for building and training **fully differentiable evolutionary and swarm-based optimisation algorithms** using PyTorch. It implements reparameterised and gradient-friendly versions of classical algorithms like **Genetic Algorithms (GA)**, **Differential Evolution (DE)**, **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)**, and **Particle Swarm Optimization (PSO)**.

By replacing discrete selection and mutation steps with **continuous, differentiable alternatives**—such as soft selection via Gumbel-Softmax, blend crossover, and reparameterised Gaussian mutation—EvoGrad enables **end-to-end backpropagation** through population dynamics. Each algorithm supports learnable hyperparameters that can be optimised alongside the solution candidates via Gradient Descent-based optimisers.

EvoGrad is ideal for researchers exploring **meta-optimisation**, **differentiable programming**, or **learning-to-optimise** frameworks.
