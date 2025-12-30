# type: ignore
import quimb.tensor as qb
import torch
import numpy as np

class BatchMovingEnvironment(qb.MovingEnvironment):
    """
    Extended MovingEnvironment that handles batch dimensions AND output dimensions.
    """
    
    def __init__(self, tn, begin, bsz, output_dims=None, batch_inds=None, **kwargs):
        self.batch_inds = set(batch_inds) if batch_inds is not None else set()
        self.output_dims = set(output_dims) if output_dims is not None else set()
        super().__init__(tn, begin, bsz, **kwargs)

    def init_non_segment(self, start, stop):
        """Initialize non-segment part with proper dtype handling for torch/numpy."""
        self.tnc = self.tn.copy(virtual=True)

        if not self.segmented and not self.cyclic:
            sample_data = next(iter(self.tn.tensors)).data
            is_torch = isinstance(sample_data, torch.Tensor)
            dtype = sample_data.dtype if is_torch else self.tn.dtype

            if is_torch:
                d = torch.tensor(1.0, dtype=dtype)
            else:
                d = np.array(1.0).astype(dtype)

            self.tnc |= qb.Tensor(data=d, tags="_LEFT")
            self.tnc |= qb.Tensor(data=d, tags="_RIGHT")
            return

        super().init_non_segment(start, stop)

    def _get_inds_to_keep(self, tn, active_tensors, site_tag=None):
        """
        Helper to determine correct output indices for contractions.
        Preserves Bonds, Global Outer, Batch, and Output indices.
        
        Args:
            tn: The tensor network
            active_tensors: Tensors being contracted
            site_tag: Tag of the current site (to preserve bonds to it)
        """
        active_inds = set().union(*(t.inds for t in active_tensors))
        all_inds = set().union(*(t.inds for t in tn))
        passive_inds = all_inds - active_inds 
        
        # 1. Keep bonds (between active and other passive tensors)
        inds_to_keep = active_inds.intersection(passive_inds)
        
        # 2. Keep bonds to current site (critical for MovingEnvironment!)
        if site_tag is not None:
            site_tensors = tn.select(site_tag)
            site_inds = set().union(*(t.inds for t in site_tensors))
            bonds_to_site = active_inds.intersection(site_inds)
            inds_to_keep.update(bonds_to_site)
        
        # 3. Keep global outer
        global_outer = set(tn.outer_inds())
        inds_to_keep.update(active_inds.intersection(global_outer))
        
        # 4. Keep Batch and Output
        if self.batch_inds:
            inds_to_keep.update(self.batch_inds.intersection(active_inds))
        if self.output_dims:
            inds_to_keep.update(self.output_dims.intersection(active_inds))
            
        return tuple(inds_to_keep)

    def init_segment(self, begin, start, stop):
        if (start >= self.L) or (stop < 0):
            start, stop = start % self.L, stop % self.L

        self.segment = range(start, stop)
        self.init_non_segment(start, stop + self.bsz // 2)

        if begin == "left":
            tags_initial = ["_RIGHT"] + [self.site_tag(stop - 1 + b) for b in range(self.bsz)]
            self.envs = {stop - 1: self.tnc.select(tags_initial, which='any')}

            for i in reversed(range(start, stop - 1)):
                self.envs[i] = self.envs[i + 1].copy(virtual=True)
                
                # Absorb site
                self.envs[i] |= self.tnc.select(self.site_tag(i))
                
                # Contract Right Boundary + Site(i+bsz)
                tags_to_contract = ("_RIGHT", self.site_tag(i + self.bsz))
                active_tensors = self.envs[i].select(tags_to_contract, which='any')
                
                # Pass site_tag so we preserve bonds to current site i
                out_inds = self._get_inds_to_keep(self.envs[i], active_tensors, site_tag=self.site_tag(i))
                self.envs[i].contract(tags_to_contract, output_inds=out_inds, inplace=True)

            self.envs[start] |= self.tnc["_LEFT"]
            self.pos = start
            
        elif begin == "right":
            tags_initial = ["_LEFT"] + [self.site_tag(start + b) for b in range(self.bsz)]
            self.envs = {start: self.tnc.select(tags_initial, which='any')}

            for i in range(start + 1, stop):
                self.envs[i] = self.envs[i - 1].copy(virtual=True)
                
                # Absorb site
                self.envs[i] |= self.tnc.select(self.site_tag(i + self.bsz - 1))
                
                # Contract Left Boundary + Site(i-1)
                tags_to_contract = ("_LEFT", self.site_tag(i - 1))
                active_tensors = self.envs[i].select(tags_to_contract, which='any')
                
                # Pass site_tag so we preserve bonds to current site i+bsz-1
                out_inds = self._get_inds_to_keep(self.envs[i], active_tensors, site_tag=self.site_tag(i + self.bsz - 1))
                self.envs[i].contract(tags_to_contract, output_inds=out_inds, inplace=True)

            self.envs[i] |= self.tnc["_RIGHT"]
            self.pos = stop - 1

    def move_right(self):
        if (not self.cyclic) and (self.pos + 1 not in self.segment):
            raise ValueError("For OBC, ``0 <= position <= n - bsz``.")

        i = (self.pos + 1) % self.L
        if i not in self.segment:
            self.init_segment("left", i, i + self._ssz)
        else:
            self.pos = i

        i0 = self.segment.start
        if i >= i0 + 1:
            # === FIX: Safely Clean Stale Left Environment ===
            # Check existence first to avoid KeyError, then remove using inverse select
            if "_LEFT" in self.envs[i].tags:
                self.envs[i] = self.envs[i].select("_LEFT", which="!any")

            # === Update Logic ===
            new_left = self.envs[i - 1].select(["_LEFT", self.site_tag(i - 1)], which="any")
            # Pass self.envs[i] so we can find site i tensors
            out_inds = self._get_inds_to_keep(self.envs[i], new_left, site_tag=self.site_tag(i)) 
            
            # Use 'all' contraction with explicit output indices
            contracted_left = new_left.contract(all, output_inds=out_inds)
            
            self.envs[i] |= contracted_left

    def move_left(self):
        if (not self.cyclic) and (self.pos - 1 not in self.segment):
            raise ValueError("For OBC, ``0 <= position <= n - bsz``.")

        i = (self.pos - 1) % self.L

        if i not in self.segment:
            self.init_segment("right", i - self._ssz + 1, i + 1)
        else:
            self.pos = i

        i0 = self.segment.start
        if i < len(self.segment) + i0 - 1:
            # === FIX: Safely Clean Stale Right Environment ===
            if "_RIGHT" in self.envs[i].tags:
                self.envs[i] = self.envs[i].select("_RIGHT", which="!any")

            # === Update Logic ===
            new_right = self.envs[i + 1].select(["_RIGHT", self.site_tag(i + self.bsz)], which="any")
            # Pass self.envs[i] so we can find site i tensors
            out_inds = self._get_inds_to_keep(self.envs[i], new_right, site_tag=self.site_tag(i))
            
            # Use 'all' contraction with explicit output indices
            contracted_right = new_right.contract(all, output_inds=out_inds)
            
            self.envs[i] |= contracted_right
