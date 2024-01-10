import torch

from pytorch_tabnet.tab_model import TabNetClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tabnet(args, cat_idxs, cat_dims) : 

    return TabNetClassifier(cat_idxs = cat_idxs, 
                            cat_dims = cat_dims,
                            cat_emb_dim = 10, 
                            seed = args.seed,
                            optimizer_fn = torch.optim.Adam,
                            optimizer_params = {"lr" : 1e-2},
                            scheduler_params = {"step_size" : 50,
                                                "gamma" : 0.9},
                            scheduler_fn = torch.optim.lr_scheduler.StepLR,
                            mask_type = 'sparsemax', # "sparsemax", entmax
                            device_name = device.type
                            )