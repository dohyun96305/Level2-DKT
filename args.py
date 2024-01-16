import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir",
        default = "/data/ephemeral/data/",
        type = str,
        help = "data directory",
    )

    parser.add_argument("--seed", default = 49, type = int, help = "seed")
    parser.add_argument("--valid_size", default = 0.2, type = float, help = "Train/Valid split size")

    parser.add_argument("--n_epochs", default = 3, type = int, help = "")
    parser.add_argument("--batch_size", default = 1024, type = int, help = "")
    parser.add_argument("--cat_emb_dim", default = 10, type = int, help = "")

    parser.add_argument("--model", default = 'tabnet', type = str, help = "")
    parser.add_argument("--model_dir", default = "./models/", type = str, help = "")
    parser.add_argument("--submit_dir", default = "./submits/", type = str, help = "")

    args = parser.parse_args()

    return args
