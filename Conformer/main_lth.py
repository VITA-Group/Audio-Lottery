# Pytorch
import torch

# Functions and Utils
from functions import *
from utils.preprocessing import *
from utils.pruning_utils import prune_main, check_sparsity

# Other
import json
import argparse
import os
import copy


def main(rank, args):
    # Process rank
    args.rank = rank

    # Distributed Computing
    if args.distributed:
        torch.cuda.set_device(args.rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size,
                                             rank=args.rank)

    # Load Config
    with open(args.config_file) as json_config:
        config = json.load(json_config)

    # Device
    device = torch.device("cuda:" + str(args.rank) if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    # Create Tokenizer
    if args.create_tokenizer:

        if args.rank == 0:
            print("Creating Tokenizer")
            create_tokenizer(config["training_params"], config["tokenizer_params"])

        if args.distributed:
            torch.distributed.barrier()

    # Create Model
    model = create_model(config)

    # Parallel Strategy
    if args.parallel and not args.distributed:
        print("Parallelize model on", torch.cuda.device_count(), "GPUs")
        model.parallel_strategy()
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Load Model
    if args.start_pt is not None:
        original_model_path = model.lth_load(config["training_params"]["callback_path"], args.start_pt, args.initial_epoch)
    else:
        args.start_pt = 0
        original_model_path = None

    # Model Summary
    if args.rank == 0:
        model.summary(show_dict=args.show_dict)

    # Distribute Strategy
    if args.distributed:
        if args.rank == 0:
            print("Parallelize model on", args.world_size, "GPUs")
        model.distribute_strategy(args.rank)

    
    # Prepare Dataset
    if args.prepare_dataset:

        if args.rank == 0:
            print("Preparing dataset")
            prepare_dataset(config["training_params"], config["tokenizer_params"], model.tokenizer, config["encoder_params"])

        if args.distributed:
            torch.distributed.barrier()

    # Load Dataset
    dataset_train, dataset_val = load_datasets(config["training_params"], config["tokenizer_params"], args)

    ###############################################################################
    # Modes
    ###############################################################################

    # Training
    if args.mode.split("-")[0] == "training":
        for pt in range(args.start_pt, config["pruning_params"]["prune_times"] + 1):
            callback_path = os.path.join(config["training_params"]["callback_path"], "prune_{}".format(pt))
            os.makedirs(callback_path, exist_ok=True)

            if args.rank == 0 and pt == 0:
                if not args.initial_epoch:
                    original_model_path = os.path.join(callback_path, "checkpoints_0.ckpt")
                    model.save(original_model_path)
            if args.rank == 0 and pt > 0:
                if (check_sparsity(model) < 100.0) and (args.start_pt is not None) and (args.initial_epoch is not None) and (pt == args.start_pt):
                    print(f'Initializing from sparsity {check_sparsity(model)}, epoch {args.initial_epoch}.')
                    print('skip prune and rewind since initialing from a checkpoint in current sparsity.')
                else:
                    model.scheduler.model_step = -1
                    model.scheduler.step()
                    model.prune_and_rewind(original_model_path, config["pruning_params"]["prune_percentage"])
                    args.initial_epoch = 0

            model.fit(dataset_train,
                      config["training_params"]["epochs"],
                      dataset_val=dataset_val,
                      val_steps=args.val_steps,
                      verbose_val=args.verbose_val,
                      initial_epoch=int(args.initial_epoch) if args.initial_epoch is not None else 0,
                      callback_path=callback_path,
                      steps_per_epoch=args.steps_per_epoch,
                      mixed_precision=config["training_params"]["mixed_precision"],
                      accumulated_steps=config["training_params"]["accumulated_steps"],
                      saving_period=args.saving_period,
                      val_period=args.val_period)

    # Evaluation
    elif args.mode.split("-")[0] == "validation" or args.mode.split("-")[0] == "test":
        # Greedy Search Evaluation
        if args.greedy or model.beam_size is None:

            if args.rank == 0:
                print("Greedy Search Evaluation")
            wer, _, _, _ = model.evaluate(dataset_val, eval_steps=args.val_steps, verbose=args.verbose_val, beam_size=1,
                                          eval_loss=args.eval_loss)

            if args.rank == 0:
                print("Geady Search WER : {:.2f}%".format(100 * wer))

        # Beam Search Evaluation
        else:

            if args.rank == 0:
                print("Beam Search Evaluation")
            wer, _, _, _ = model.evaluate(dataset_val, eval_steps=args.val_steps, verbose=args.verbose_val,
                                          beam_size=model.beam_size, eval_loss=False)

            if args.rank == 0:
                print("Beam Search WER : {:.2f}%".format(100 * wer))

    # Eval Time
    elif args.mode.split("-")[0] == "eval_time":

        print("Model Eval Time")
        inf_time = model.eval_time(dataset_val, eval_steps=args.val_steps, beam_size=1,
                                   rnnt_max_consec_dec_steps=args.rnnt_max_consec_dec_steps, profiler=args.profiler)
        print("eval time : {:.2f}s".format(inf_time))

    elif args.mode.split("-")[0] == "eval_time_encoder":

        print("Encoder Eval Time")
        enc_time = model.eval_time_encoder(dataset_val, eval_steps=args.val_steps, profiler=args.profiler)
        print("eval time : {:.2f}s".format(enc_time))

    elif args.mode.split("-")[0] == "eval_time_decoder":

        print("Decoder Eval Time")
        dec_time = model.eval_time_decoder(dataset_val, eval_steps=args.val_steps, profiler=args.profiler)
        print("eval time : {:.2f}s".format(dec_time))

    # Destroy Process Group
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="configs/EfficientConformerCTCLargeLTH.json",
                        help="Json configuration file containing model hyperparameters")
    parser.add_argument("-m", "--mode", type=str, default="training",
                        help="Mode : training, validation-clean, test-clean, eval_time-dev-clean, ...")
    parser.add_argument("-d", "--distributed", action="store_true", help="Distributed data parallelization")
    parser.add_argument("-i", "--initial_epoch", type=str, default=None, help="Load model from checkpoint")
    parser.add_argument("-p", "--prepare_dataset", action="store_true", help="Prepare dataset for training")
    parser.add_argument("-j", "--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--create_tokenizer", action="store_true", help="Create model tokenizer")
    parser.add_argument("--batch_size_eval", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--verbose_val", action="store_true", help="Evaluation verbose")
    parser.add_argument("--val_steps", type=int, default=None, help="Number of validation steps")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Number of steps per epoch")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of available GPUs")
    parser.add_argument("--cpu", action="store_true", help="Load model on cpu")
    parser.add_argument("--show_dict", action="store_true", help="Show model dict summary")
    parser.add_argument("--parallel", action="store_true", help="Parallelize model using data parallelization")
    parser.add_argument("--rnnt_max_consec_dec_steps", type=int, default=None,
                        help="Number of maximum consecutive transducer decoder steps during inference")
    parser.add_argument("--eval_loss", action="store_true", help="Compute evaluation loss during evaluation")
    parser.add_argument("--greedy", action="store_true", help="Proceed to a greedy search evaluation")
    parser.add_argument("--saving_period", type=int, default=1, help="Model saving every 'n' epochs")
    parser.add_argument("--val_period", type=int, default=1, help="Model validation every 'n' epochs")
    parser.add_argument("--profiler", action="store_true", help="Enable eval time profiler")
    parser.add_argument("--start_pt", type=int, default=None, help="Start pruning percentage")

    # Parse Args
    args = parser.parse_args()

    # Run main
    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        torch.multiprocessing.spawn(main, nprocs=args.world_size, args=(args,))
    else:
        main(0, args)
