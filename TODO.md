There is a problem due to some version mismatch etc...

In outputs, some runs that are singular are saved with the val_quality = null

/home/nicci/Desktop/remote/GTN/outputs/ntn/breast/MMPO2_rg5_init0.1/L3_bd4_seed47311/results.json

Can we make a script that check all the results.json and verify that the reported val_quality is the correct one? which would be the highest val quality, and then all the other parameters like train_loss train_quality and val_loss at that epoch of the best val_quality
