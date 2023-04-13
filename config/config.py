import argparse

BATCH_SIZE = 4

DATA_PATH = "./data/"



def get_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    
    parser.add_argument("--task", type=str, help="Name of this training")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Dataset root path.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training. ")
    parser.add_argument("--debug_mode", dest='debug_mode', action='store_true',  help="If debug mode, load less data.")    
    # parser.add_argument("--gamma", dest='gamma', action='store_true', help="Use gamma compression for raw data.")
    # parser.add_argument("--camera", type=str, default="NIKON_D700", choices=["NIKON_D700", "Canon_EOS_5D"], help="Choose which camera to use. ")    
    parser.add_argument("--rgb_weight", type=float, default=1, help="Weight for rgb loss. ")
    
    parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save checkpoint. ")
    parser.add_argument("--resume", dest='resume', action='store_true',  help="Resume training. ")
    parser.add_argument("--loss", type=str, default="L1", choices=["L1", "L2"], help="Choose which loss function to use. ")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--aug", dest='aug', action='store_true', help="Use data augmentation.")
    parser.add_argument("--resolution", default=1024, type=int, help="crop size.")
    parser.add_argument("--block_num", default=8, type=int, help="number of invertible blocks.")
    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--rgb_root", type=str, required=True)
    # parser.add_argument("--meta_root", type=str, required=True)
    parser.add_argument("--world_size", type=int, default=8, help="number of gpus in total")

    args = parser.parse_args()
    return args
