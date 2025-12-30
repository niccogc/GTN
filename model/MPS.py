# type: ignore
import quimb.tensor as qt
from model.NTN import NTN
from model.batch_moving_environment import BatchMovingEnvironment

class CMPO2_NTN(NTN):
    def __init__(self, *args, cache_environments=False, **kwargs):
        self.cache_environments = cache_environments
        self._env_cache = {} 
        self._tag_to_position = None 
        self._cache_hits = 0
        self._cache_misses = 0
        super().__init__(*args, **kwargs)

    def _get_tag_to_position_map(self):
        if self._tag_to_position is not None:
            return self._tag_to_position
        
        trainable_nodes = self._get_trainable_nodes()
        self._tag_to_position = {tag: i for i, tag in enumerate(trainable_nodes)}
        return self._tag_to_position
    
    def clear_cache(self):
        self._env_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self):
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total,
            'hit_rate': hit_rate
        }

    def _batch_environment(self, inputs, tn: qt.TensorNetwork, target_tag: str,
                           sum_over_batch: bool = False, sum_over_output: bool = False) -> qt.Tensor:
        
        if not self.cache_environments:
            return super()._batch_environment(inputs, tn, target_tag, 
                                             sum_over_batch=sum_over_batch,
                                             sum_over_output=sum_over_output)
        # --- Caching Logic ---
        
        # 1. Cache Key
        if isinstance(inputs, list):
            cache_key = id(inputs[0])
        else:
            cache_key = id(list(inputs)[0])
        
        # 2. Get/Create Environment
        if cache_key not in self._env_cache:
            full_tn = tn.copy()
            for t in inputs:
                full_tn.add_tensor(t)
            
            self._env_cache[cache_key] = BatchMovingEnvironment(
                full_tn,
                begin='left',
                bsz=1,
                batch_inds=[self.batch_dim],
                output_dims=set(self.output_dims) # Must pass output_dims here
            )
            self._cache_misses += 1
        else:
            self._cache_hits += 1
        
        # 1. Retrieve Env Object
        env_obj = self._env_cache[cache_key]
        
        # 2. Move to Site
        tag_to_pos = self._get_tag_to_position_map()
        site_idx = tag_to_pos[target_tag]
        env_obj.move_to(site_idx)
        
        # 3. Get Base Environment
        # env_obj() returns: _LEFT + _RIGHT + SAME_SITE tensors
        # SAME_SITE includes all tensors at this site (MPS tensors + inputs)
        base_env = env_obj() 
        
        # 4. Create hole by deleting only the target
        # We copy to avoid modifying the cached environment
        final_env = base_env.copy()
        final_env.delete(target_tag)
        
        # 6. Determine Output Indices
        outer_inds = final_env.outer_inds()
        inds_to_keep = list(outer_inds)
        
        # Safety: Ensure batch/out are kept if present in the final assembly
        all_inds = set().union(*(t.inds for t in final_env))
        
        if self.batch_dim in all_inds and self.batch_dim not in inds_to_keep:
            inds_to_keep.append(self.batch_dim)
            
        for out_dim in self.output_dimensions:
            if out_dim in all_inds and out_dim not in inds_to_keep:
                inds_to_keep.append(out_dim)

        # Handle Summing
        if sum_over_batch and self.batch_dim in inds_to_keep:
            inds_to_keep.remove(self.batch_dim)
        
        if sum_over_output:
            for out_dim in self.output_dimensions:
                if out_dim in inds_to_keep:
                    inds_to_keep.remove(out_dim)
        
        # 7. Contract (use 'all' for hyper-edge networks)
        return final_env.contract(all, output_inds=inds_to_keep)    
    
    def _get_trainable_nodes(self):
        from model.NTN import NOT_TRAINABLE_TAG
        
        trainable_tags = []
        for tensor in self.tn:
            if NOT_TRAINABLE_TAG in tensor.tags:
                continue
            
            # Filter standard MPS tags
            valid_tags = [tag for tag in tensor.tags 
                         if not tag.startswith('I') or not tag[1:].isdigit()
                         if not 'BLOCK' in tag]
            
            if valid_tags:
                trainable_tags.append(valid_tags[0])
        
        return trainable_tags
    
    def _get_column_structure(self):
        trainable_nodes = self._get_trainable_nodes()
        columns = {}
        for tag in trainable_nodes:
            col_idx = int(tag.split('_')[0])
            if col_idx not in columns:
                columns[col_idx] = []
            columns[col_idx].append(tag)
        
        max_col = max(columns.keys())
        return [columns[i] for i in range(max_col + 1)]
    
    def fit(self, n_epochs=1, regularize=True, jitter=1e-6, verbose=True, eval_metrics=None):
        if not self.cache_environments:
            return super().fit(n_epochs=n_epochs, regularize=regularize, 
                             jitter=jitter, verbose=verbose, eval_metrics=eval_metrics)
        
        from model.NTN import REGRESSION_METRICS, print_metrics
        
        if eval_metrics is None:
            eval_metrics = REGRESSION_METRICS
        
        if not isinstance(jitter, list):
            jitter = [jitter] * n_epochs
        
        columns = self._get_column_structure()
        n_cols = len(columns)
        
        if verbose:
            print(f"Starting Fit: {n_epochs} epochs (cached sweeping).")
            print(f"Column structure: {columns}")
        
        # Initial Eval
        scores = self.evaluate(eval_metrics)
        if verbose:
            print(f"Init    | ", end="")
            print_metrics(scores)
        
        for epoch in range(n_epochs):
            # Forward
            for col_idx in range(n_cols):
                for node_tag in columns[col_idx]:
                    self.update_tn_node(node_tag, regularize, jitter[epoch])
            
            # Backward
            for col_idx in range(n_cols - 2, 0, -1):
                for node_tag in columns[col_idx]:
                    self.update_tn_node(node_tag, regularize, jitter[epoch])
            
            self.clear_cache()
            
            scores = self.evaluate(eval_metrics)
            if verbose:
                print(f"Epoch {epoch+1} | ", end="")
                print_metrics(scores)
        
        if verbose:
            stats = self.get_cache_stats()
            print(f"\nCache stats: {stats['hits']} hits, {stats['misses']} misses")
        
        return scores
