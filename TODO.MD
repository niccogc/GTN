So,
First of all the configs file are wrong.
for gtn it needs 200 epochs.

Also, IT SHOULDNT BE A SEPARATE RUN AND CYCLE FOR DIMENSION REDUCTION FOR LMPO2, IT SHOULD BE A PARAMETER OF THE MAIN GRID SEARCH, WITHIN THE SAME FILE FOR A DATASET AND IT JUST GETS PASSED THE ARGUMENT IF THE MODEL IS LMPO2 OR LMPO2 TYPEI.

Initialization. It should be normalied. can we first try that before introducing a parameter like initialization strenght??

Aim seems not to register run, when aim register it just at the end of a .sh

lso some run it is said, "Already done" can you verify how it works the check for already run runs? and clear the cache

