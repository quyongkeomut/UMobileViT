alpha: float =  0.5
out_channel = [2, 2]
patch_size = (2,2)
num_epochs: int = 50
batch_size: int = 24
optimizer: str = "adamw"

lr: float = 5e-4
optim_args = {
        "lr": lr,
        # "momentum": momentum,
        # "nesterov": nesterov 
    }
device = "cuda"
check_point = None