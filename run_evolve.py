from utils.evolve import evol
from utils import set_seed, check_dir, print_config
import config as cfg_ori
import argparse


def run(cfg):
    evolver = evol(cfg)
    evolver.evolve()


if __name__ == '__main__':
    set_seed(123)
    parser = argparse.ArgumentParser(description="Evolve")
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--target_round', type=int, default=-1)
    parser.add_argument('--model_name', type=str, default='')
    args = parser.parse_args()
    # update cfg
    if args.num_workers != -1:
        cfg_ori.num_workers = args.num_workers

    if args.target_round != -1:
        cfg_ori.target_round = args.target_round

    if args.model_name != '':
        cfg_ori.model_name = args.model_name

    for path in [cfg_ori.seed_path, cfg_ori.save_path, cfg_ori.final_data_path,
                 cfg_ori.deep_cache, cfg_ori.fuse_cache, cfg_ori.round_info,
                 cfg_ori.comm_to_cal, cfg_ori.comm_gen_resp, cfg_ori.comm_cal_done]:
        check_dir(path)

    print_config(cfg_ori, ['prefix'])

    run(cfg_ori)
