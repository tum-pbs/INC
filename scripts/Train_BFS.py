import os, glob, re
import sys

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lib.util.logging import setup_run, get_logger, close_logging # logging and output

def get_model_dir(id,task="BFS",logging=True):
    base_path = f"./INC_Data/{task}/Results"
    search_pattern = os.path.join(base_path, "*", f"*{id}*")
    matched_paths = glob.glob(search_pattern)
    if not matched_paths:
        raise FileNotFoundError(f"No directory found matching pattern: {search_pattern}")
    if len(matched_paths) > 1:
        raise ValueError(f"Multiple directories match the pattern: {matched_paths}")
    path = matched_paths[0]
    if logging:
        from lib.util.logging import setup_logging
        setup_logging(os.path.join(path, "test_log"), console=True, debug=False)
    return path

def naming(train_params,sim_params):
    return f"mstep{train_params.mstep}_substep{sim_params.substeps}_LR{train_params.lr}_WD{train_params.weight_decay}"

def main(args):
    index_start = 1
    index_end = 2
    mstep_list=[2,8,16][index_start:index_end]
    lr_list=[1e-4,1e-4,1e-5][index_start:index_end]
    weight_decay_list = [1e-1,1e-1,1e-1][index_start:index_end]
    method = args.method
    model_type = args.model_type
    substeps = args.substeps if args.substeps !=-1 else  "ADAPTIVE"
    model_id = args.model_id
    method_base= f"{method}_{model_type}"
    task = "BFS"
    if args.mode == "train":
        last_run_dir, model_dir,test_after_train = None, None, True
        for i in range(len(mstep_list)):
            mstep=mstep_list[i]
            train_params = config.BFSTrainParams(mstep=mstep,lr=lr_list[i], model_type=model_type,method=method,  weight_decay = weight_decay_list[i])
            sim_params = config.BFSSimParams(Re=train_params.Re, s=train_params.geo_feature, downsample_factor=4,substeps=substeps)
            train_params.valid_steps = train_params.valid_steps
            run_dir = setup_run(f"./INC_Data/{task}/Results/{method_base}", name=naming(train_params,sim_params))
            # if load model instead of training from scratch
            model_dir=get_model_dir(model_id,logging=False) if model_id is not None else None
            if last_run_dir is not None:
                time_stamp=re.search(r'/(\d{6}-\d{6})_', last_run_dir).group(1)
                model_dir = get_model_dir(time_stamp,logging=False)
                LOG.info(f"Model directory: {model_dir}")
            trainer = Trainer(run_dir=run_dir,sim_params=sim_params, train_params=train_params,model_dir=model_dir,LOG=LOG, task=task)
            LOG.info(f"Parsed arguments: {vars(args)}")
            trainer.train()
            last_run_dir = run_dir
            close_logging()
            if test_after_train and mstep>=4:
                time_stamp=re.search(r'/(\d{6}-\d{6})_', last_run_dir).group(1)
                model_dir=get_model_dir(time_stamp)
                test_params = config.BFSTrainParams(mstep=None,model_type=model_type,method=method)
                test_params.test_steps = 1200
                trainer = Trainer(run_dir=model_dir,sim_params=sim_params, train_params=test_params, model_dir=model_dir,  LOG=LOG, task=task)
                trainer.start_points = range(3000, 4500, 100)
                trainer.test()
                close_logging()
                
    if args.mode == "test":
        model_dir=get_model_dir(model_id)
        test_params = config.BFSTrainParams(mstep=None,model_type=model_type,method=method)
        sim_params = config.BFSSimParams(Re=test_params.Re, s=test_params.geo_feature, downsample_factor=4)
        sim_params.substeps = substeps
        test_params.test_steps = 1200
        trainer = Trainer(run_dir=model_dir,sim_params=sim_params, train_params=test_params, model_dir=model_dir,  LOG=LOG, task=task)
        trainer.start_points = range(3000, 4500, 100)
        LOG.info(f"Parsed arguments: {vars(args)}")
        load_check_point=False
        if load_check_point: 
            for i in range(3,6):
                try:
                    trainer.test(checkID=i)
                except:
                    print(f"Check point {i} failed")
        else:
            trainer.test()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BFS Simulation with configurable parameters.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument("--method", type=str, default="INC", help="Method to use")
    parser.add_argument("--model_type", type=str, default="SmallCNN", help="Model type")
    parser.add_argument("--substeps", type=int, default=1, help="Number of substeps")
    parser.add_argument('--model_id', type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import torch
    assert torch.cuda.is_available()
    from config import config_2d as config
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from trainer.trainer_2d import Trainer

    print(f"Using GPU: {args.gpu_id}")
    LOG = get_logger("Main")
    main(args)
    close_logging()