We want, in this repo to implement a new model, so first understand how a model is constructed and then we construct two new models.
TNML-F TNML-P. check run.py, model/standard , NTN GTN.
The new model is a more standard MPS model, but basically you can see that the existing models are polynomial models where L is the degree of the polynomial. Until now the previous model used L blocks (nodes) with repeating the same input, hence creating a L degree polnmial.
Now TNML-P, will be a polynomial, but now model.L+1 is the dimension of the physical dimension, input dimension. and we have an MPS with as many nodes as the features! so pretty much a standard MPS,
While TNML-F, will have the same nodes as features, but always 2 input dimension.
So how the input must be prepared? Differently from run.py, the inputs for TNML-P will be for each node the bias term 1, then x1, x1**2, ... , x1**L. and likewise for each features.
While TNML-P will simply have cos(x1*pi/2), sin(x1* pi/2) forurier features. 
The steps, 
Check how models are implemented to be consistent.
Implement the structures for TNML-P, TNML-F and test using print and checking bond matches.
Check if the existing builder functions can be used to generate the two different type of inputs. Reuse as much code as possible.
Implement the config and make it production ready that can be used with run.py and we can generate jobs, and scripts/backfill_tracking.py and check_progress.py are adapted to the new models
Activate serena project and use docs-mcp to query quimb docs
