import random
from typing import List, Dict, Any, Tuple
import quimb.tensor as qt  # Assuming quimb.tensor is available

class Inputs:
    """
    Input loader that pre-computes and stores batches in a list.
    Allows efficient multiple passes (epochs) over the data.
    """
    def __init__(self, inputs: List[Any],
                 outputs: List[Any],
                 outputs_labels: List[str],
                 input_labels: List[str],
                 batch_dim: str = "s",
                 batch_size = None):

        # 1. Configuration
        self.inputs_data = inputs
        self.outputs_data = outputs
        self.outputs_labels = outputs_labels
        self.input_labels = input_labels
        self.batch_dim = batch_dim
        
        self.batch_size = inputs[0].shape[0] if batch_size is None else batch_size
        self.samples = outputs[0].shape[0]
        self.repeated = (len(inputs) == 1)
        
        # 2. Pre-compute all batches once and store them
        # Storage format: List[Tuple(mu_tensors, prime_tensors, y_tensor)]
        self.batches = self._create_batches()

    def _create_batches(self) -> List[Tuple[List[qt.Tensor], qt.Tensor]]:
        """Generates and stores the list of all batches."""
        batches = []
        
        # Generator for raw data slices
        raw_splits = self.batch_splits(
            self.inputs_data, 
            self.outputs_data[0], 
            self.batch_size
        )

        # Process into tensors
        for input_dict, y_tensor in raw_splits:
            if self.repeated:
                mu = self.prepare_inputs_batch_repeated(input_dict)
            else:
                mu = self.prepare_inputs_batch(input_dict)
           
            batches.append((mu, y_tensor))
            
        return batches

    # --- Properties for Iteration ---

    @property
    def data_mu(self):
        """Yields only mu input tensors."""
        for mu, _ in self.batches:
            yield mu

    @property
    def data_y(self):
        """Yields only target (y) tensors."""
        for _, y in self.batches:
            yield y

    @property
    def data_mu_y(self):
        """Yields (mu inputs, target)."""
        for mu, y in self.batches:
            yield mu, y

    # --- Processing Methods ---

    def batch_splits(self, xs, y, B):
        """Generates raw dictionary/array slices."""
        s = y.shape[0]
        for i in range(0, s, B):
            tensor = qt.Tensor(
                data=y[i:i+B],
                inds=(self.batch_dim, *self.outputs_labels),
                tags={'output'}
            )
            batch = {f"{j}": x[i:i+B] for j, x in zip(self.input_labels[:len(xs)], xs)}
            yield batch, tensor

    def prepare_inputs_batch_repeated(self, input_data: Dict[str, Any]) -> List[qt.Tensor]:
        input_indices = self.input_labels
        single_key = list(input_data.keys())[0]
        data = input_data[single_key]
        
        tensors = []
        for input_idx in input_indices:
            # Mu tensor
            tensor = qt.Tensor(data=data, inds=(self.batch_dim, input_idx), tags={f'input_{input_idx}'})
            tensors.append(tensor)
        
        return tensors

    def prepare_inputs_batch(self, input_data: Dict[str, Any]) -> List[qt.Tensor]:
        """
        Returns:
            tensors: List of tensors for mu network [x1, x2...]
        """
        tensors = []
        for k, v in input_data.items():
            
            # Mu tensor
            tensor = qt.Tensor(data=v, inds=(self.batch_dim, k), tags={f'input_{k}'})
            tensors.append(tensor)
        
        return tensors

    def shuffle(self):
        """Shuffles the internal list of batches in-place."""
        random.shuffle(self.batches)

    def __len__(self):
        """Returns the number of batches."""
        return len(self.batches)

    def __getitem__(self, idx):
        """
        Returns the raw batch at the specified index.
        
        Returns:
            Tuple: (mu_tensors, prime_tensors, y_tensor)
        
        Note: To get full sigma inputs from this, you must concatenate mu + prime.
        """
        return self.batches[idx]

    def __str__(self):
        """Summary of the loader structure."""
        if not self.batches:
            return ">>> InputLoader (Empty)"

        # Peek at the first stored batch
        mu, y = self.batches[0]
        
        mu_inds = [list(t.inds) for t in mu]
        
        header = (
            f"\n>>> InputLoader Summary (Batch Size: {self.batch_size}, "
            f"Samples: {self.samples}, Batches: {len(self.batches)})\n"
            f"{'TYPE':<8} | {'SHAPE':<15} | {'INDICES'}\n"
            f"{'-'*60}\n"
        )
        
        row_y = f"{'Target':<8} | {str(y.shape):<15} | {y.inds}\n"
        row_mu = f"{'Mu':<8} | {str(mu[0].shape):<15} | {mu_inds} ... ({len(mu)} tensors)\n"
        
        return header + row_y + row_mu
