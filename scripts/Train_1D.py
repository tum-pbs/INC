import os
import sys

def main(args):
    # Import necessary modules
    import torch
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.tensorboard import SummaryWriter
    from trainer.trainer_1d import (
        CorrectionTrainer, get_task_config, init_dataloader, 
        load_model_dir, get_path, check_dt
    )
    from lib.util.logging import setup_run, get_logger, close_logging
    
    # Load the dataset
    train_param = get_task_config(args)["train_param"]
    path_dict = get_path(train_param.down_resolution, task=args.task)
    data_dict = {split: torch.load(path_dict[split]) for split in ("train", "valid", "test")}
    train_param.metadata = data_dict["train"]['metadata']
    train_param.init_model() if train_param.model is None else None
    task_dir = f"INC_Data/{args.task}/Results/"
    
    if args.mode == "train":
        LOG = get_logger("Train")
        method_base = f"{train_param.model_type}_Corr-{train_param.correction_term}"
        case_name = f"mstep{train_param.mstep}_dt{train_param.dt}_lr{train_param.lr}_wd{train_param.weight_decay}_gp{train_param.lambda_gp}_spec{train_param.lambda_spectrum}_batch{train_param.batch_size}_ADP{train_param.adaptive_CFL}_seed{train_param.seed}_valid{train_param.valid_step}"
        run_dir = setup_run(f"{task_dir}{method_base}/Res_{train_param.down_resolution}", name=case_name)
        writer_log_dir = os.path.normpath(os.path.join(f"logs/{args.mode}", run_dir))
        os.makedirs(writer_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=writer_log_dir)
        
        # Initialize components
        if args.model_id is not None:
            model_path, model_dir = load_model_dir(train_param, run_id=args.model_id, task_dir=task_dir, LOG=LOG, mode=args.mode)
            train_param.model.load_state_dict(torch.load(model_dir, map_location=train_param.device))
            train_param.dt = check_dt(train_param, LOG, model_path)
        optimizer = Adam(train_param.model.parameters(), lr=train_param.lr, weight_decay=train_param.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=train_param.lr_decay, patience=train_param.lr_decay_patience, verbose=True, min_lr=1e-7)
        trainer = CorrectionTrainer(optimizer, scheduler, train_param, writer, LOG, run_dir)
        train_loader, valid_loader, test_loader = init_dataloader(data_dict, train_param)
        LOG.info(f"Parameters: {train_param.__dict__}")
        LOG.info(f"Parsed arguments: {vars(args)}")
        total_params = sum(p.numel() for p in train_param.model.parameters())
        trainable_params = sum(p.numel() for p in train_param.model.parameters() if p.requires_grad)
        LOG.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
        for epoch in range(train_param.epochs):
            train_loss = trainer.train(train_loader, epoch)
            valid_loss = trainer.evaluate(valid_loader, epoch)
            if trainer._save_checkpoint(epoch, train_loss, valid_loss):
                break
            LOG.info(f"Epoch {epoch}: Train loss: {train_loss:.6f}, Valid loss: {valid_loss:.6f}")
        trainer.test(test_loader)
    
    elif args.mode == "test":
        LOG = get_logger("Test")
        run_dir, best_model_path = load_model_dir(train_param, run_id=args.model_id, task_dir=task_dir, LOG=LOG, mode=args.mode)
        train_param.dt = check_dt(train_param, LOG, run_dir)
        writer_log_dir = os.path.normpath(os.path.join(f"logs/{args.mode}", run_dir + f"/Step{train_param.test_steps}"))
        os.makedirs(writer_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=writer_log_dir)
        trainer = CorrectionTrainer(None, None, train_param, writer, LOG, run_dir)
        _, _, test_loader = init_dataloader(data_dict, train_param)
        LOG.info(f"Parsed arguments: {vars(args)}")
        trainer.test(test_loader, best_model_path)
    
    writer.close()
    close_logging()

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    import argparse
    parser = argparse.ArgumentParser(description='Train/test correction model.')
    parser.add_argument('--mode', choices=['train', 'test'], default='test')
    parser.add_argument('--down_ratio', type=int, choices=[4, 8, 16, 32], default=8)
    parser.add_argument('--model_id', type=str, default="251016-174730")
    parser.add_argument('--task', type=str, default='KS')
    parser.add_argument('--correction_term', default='INC')
    parser.add_argument('--model_type', choices=[ 'FNO', 'UNet', 'DeepONet', "ResNet"], default='UNet')
    parser.add_argument('--mstep', type=int, default=5)
    parser.add_argument('--test_steps', type=int, default=5000)
    parser.add_argument('--gpu_id', type=str, default='6')
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--starts_gap', type=float, default=10)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    main(args)