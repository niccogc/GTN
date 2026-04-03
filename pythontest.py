import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

def check_config(experiment_name="uci_ntn_sweep"):
    # Initialize Hydra and point to your 'conf' folder
    # version_base=None is required for Hydra 1.2+
    with initialize(version_base=None, config_path="conf"):
        
        # This simulates: python run.py experiment=uci_ntn_sweep
        # We use '+' if experiment isn't in your main config defaults yet
        cfg = compose(config_name="config", overrides=[f"experiment={experiment_name}"])
        
        print("="*60)
        print(f" RESOLVED CONFIGURATION FOR: {experiment_name}")
        print("="*60)
        
        # 1. Print the high-level experiment name
        print(f"Experiment Name: {cfg.experiment_name}")
        
        # 2. Print Dataset Info (Checking for flattened batch_size)
        print("\n[DATASET]")
        if "dataset" in cfg:
            print(f"  Name:       {cfg.dataset.get('name')}")
            print(f"  Batch Size: {cfg.dataset.get('batch_size')}")
            print(f"  Task:       {cfg.dataset.get('task')}")
        
        # 3. Print Model Info (Checking for nested L and bond_dim)
        print("\n[MODEL]")
        if "model" in cfg:
            print(f"  Name:       {cfg.model.get('name')}")
            print(f"  L:          {cfg.model.get('L')}")
            print(f"  Bond Dim:   {cfg.model.get('bond_dim')}")
        
        # 4. Print Hydra Paths (This is where your skip_completed logic looks)
        print("\n[HYDRA PATHS]")
        try:
            # We look for hydra paths. Depending on your nesting, 
            # they might be at cfg.hydra or cfg.model.hydra
            h_cfg = cfg.get("hydra") or cfg.model.get("hydra")
            if h_cfg:
                print(f"  Sweep Dir:    {h_cfg.sweep.dir}")
                print(f"  Sweep Subdir: {h_cfg.sweep.subdir}")
            else:
                print("  ! Hydra path settings not found in root or model.")
        except Exception as e:
            print(f"  ! Error resolving paths: {e}")

        print("\n" + "="*60)
        print(" FULL YAML (For Deep Inspection)")
        print("="*60)
        print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    # You can change the experiment name here to test different files
    check_config("uci_ntn_sweep")
