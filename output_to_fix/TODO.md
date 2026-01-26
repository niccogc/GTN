These errors are inacceptable, to be fixed.

ALSO need a super small grid search test file to run all models with cuda to fastly run on the cluster and see if everything works.

**IMPORTANT**

The most inacceptable thing is that the script continued and logged the results!!

HERE ME CLEARLY. THE SCRIPT **MUST** stop for EVERY error, except singular Matrix error. WHICH you can always catch since the only time it can ever happens in the NTN inversion matrix calculation. So simply make a special class for that error which will make the script will consider as a done run if it encountered a singular matrix. ALL OTHER ERROR THE SCRIPT STOPS.
So remove all error catching for specific error, all errors must stop the script. The only error that is not considered a failure but we save the run anyway is singular matrix error.

We need also with a python code a way to remove from results ALL saved results where the error is different than the singular matrix error.

and to remove them from AIM too.

**OUTPUT INDICES**

For all models the default output index label should be on the LAST L-1 block!!
