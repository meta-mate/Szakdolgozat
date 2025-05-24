import gc
import torch

from transformers import Trainer

class SafeTrainer(Trainer):
    def training_step(self, model, inputs):
        try:
            return super().training_step(model, inputs)

        except RuntimeError as e:
            if "out of memory" in str(e):
                del inputs
                gc.collect()
                torch.cuda.empty_cache()
                raise e
