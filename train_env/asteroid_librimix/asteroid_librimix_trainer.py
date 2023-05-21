"""
 This file is copied from https://github.com/asteroid-team/asteroid/blob/master/egs/librimix/ConvTasNet/train.py
 and modified for this project needs.

 The Licence of the torch vision project is shown in: https://github.com/asteroid-team/asteroid/blob/master/LICENSE
"""
import yaml
import os
import json
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from train_env.asteroid_librimix.librimix_dataset import LibriMix
from train_env.asteroid_librimix.system import System

from asteroid.engine.optimizers import make_optimizer
from asteroid.losses import *

from utils import set_seed
from val import val_librimix
from quantization.models.load_model import create_model

def train(yml_path, device):

    # -----------------------------------
    # Read yml
    # -----------------------------------
    with open(yml_path) as f:
        conf = yaml.safe_load(f)

    work_dir, model_cfg, dataset_cfg = conf['work_dir'], conf['model'], conf['dataset']
    training_cfg, testing_cfg = conf['training'], conf['testing']

    # Ensuring training reproducibility
    seed = training_cfg.get("seed", 0)
    set_seed(seed)

    # ------------------------------------
    # Dataset 
    # ------------------------------------
    # Augmentation
    augmentation_cfg = dataset_cfg.get('augmentation',None)
    if augmentation_cfg:
        enable = augmentation_cfg.get("enable", False)
        if not enable:
            augmentation_cfg = None

    # Train dataset
    train_set = LibriMix(
        csv_dir=dataset_cfg["train_dir"],
        task=dataset_cfg["task"],
        sample_rate=dataset_cfg["sample_rate"],
        resample=dataset_cfg.get("resample",1),
        n_src=dataset_cfg["n_src"],
        segment=dataset_cfg["segment"],
        augmentation_cfg=augmentation_cfg,
    )

    # Validation dataset
    val_set = LibriMix(
        csv_dir=dataset_cfg["valid_dir"],
        task=dataset_cfg["task"],
        sample_rate=dataset_cfg["sample_rate"],
        resample=dataset_cfg.get("resample",1),
        n_src=dataset_cfg["n_src"],
        segment=dataset_cfg["segment"],
    )

    print("Training set size: {}".format(len(train_set)))
    print("Validation set size: {}".format(len(val_set)))

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        drop_last=True,
    )
    
    # ------------------------------------
    # Model
    # ------------------------------------
    float_model_cfg = model_cfg.copy()
    float_model_cfg.update({"n_splitter": model_cfg.get("n_splitter",1), "n_combiner": model_cfg.get("n_combiner",1)})
    model = create_model(float_model_cfg)
    model.train()
    
    # Load pretrained
    is_ckpt = False
    pretrained = training_cfg.get("pretrained", None)
    if pretrained is not None:
        is_ckpt = pretrained.endswith('.ckpt')
        if not is_ckpt:
            model.load_pretrain(pretrained)

    # QAT params
    quant_cfg = model_cfg['quantization']
    qat = quant_cfg.get('qat', False)
    gradient_based = quant_cfg.get("gradient_based", True)
    weight_quant, act_quant = quant_cfg.get("weight_quant", True), quant_cfg.get("act_quant", True)
    in_quant, out_quant = quant_cfg.get("in_quant", False), quant_cfg.get("out_quant", True)
    weight_n_bits, act_n_bits = quant_cfg.get('weight_n_bits', 8), quant_cfg.get('act_n_bits', 8)
    in_act_n_bits, out_act_n_bits = quant_cfg.get('in_act_n_bits', 8), quant_cfg.get('out_act_n_bits', 8)
    n_splitter_bits, n_combiner_bits = quant_cfg.get('n_splitter_bits', 8), quant_cfg.get('n_combiner_bits', 8)

    if qat:
        model.quantize_model(gradient_based=gradient_based,
                             weight_quant=weight_quant,
                             weight_n_bits=weight_n_bits,
                             act_quant=act_quant,
                             act_n_bits=act_n_bits,
                             in_quant=in_quant,
                             in_act_n_bits=in_act_n_bits,
                             out_quant=out_quant,
                             out_act_n_bits=out_act_n_bits)

    model.to(device)

    # Just after instantiating, save the args. Easy loading in the future.
    os.makedirs(work_dir, exist_ok=True)
    conf_path = os.path.join(work_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(model_cfg, outfile)
        yaml.safe_dump(dataset_cfg, outfile)
        yaml.safe_dump(training_cfg, outfile)

    # WandB
    wandbLogger = None
    if training_cfg.get("wandb", False):
        import wandb
        print("WandB is enable!")
        test_name = work_dir.split('/')[-1]
        PROJECT_NAME = model_cfg["name"] + "_" + dataset_cfg["task"]
        wandb.init(project=PROJECT_NAME, group=test_name, dir=work_dir)
        wandbLogger = WandbLogger(project=PROJECT_NAME, group=test_name, dir=work_dir)
        wandb.log({"qat":int(qat),"gradient_based":int(gradient_based),"n_splitter_bits":n_splitter_bits,"n_combiner_bits":n_combiner_bits})
        wandb.log({"weight_quant":int(weight_quant), "weight_n_bits": weight_n_bits})
        wandb.log({"act_quant":int(act_quant), "act_n_bits": act_n_bits})
        wandb.log({"in_quant":int(in_quant), "in_act_n_bits": in_act_n_bits})
        wandb.log({"out_quant":int(out_quant), "out_act_n_bits": out_act_n_bits})

    # ------------------------------------
    # Training Setup
    # ------------------------------------
    optimizer = make_optimizer(model.parameters(), **training_cfg["optim"])
    # Define scheduler
    scheduler = None
    if training_cfg.get("half_lr",False):
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=training_cfg.get("patience",5))

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=model_cfg,
        n_splitter_bits=n_splitter_bits,
        n_combiner_bits=n_combiner_bits
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(work_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True)
    callbacks.append(checkpoint)
    if training_cfg["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    trainer = pl.Trainer(
        max_epochs=training_cfg["epochs"],
        callbacks=callbacks,
        default_root_dir=work_dir,
        accelerator="gpu" if torch.cuda.is_available() and device!='cpu' else "cpu",
        strategy="ddp",
        devices="auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=wandbLogger,
    )

    # ------------------------------------
    # Training
    # ------------------------------------
    trainer.fit(system, ckpt_path=pretrained if is_ckpt else None)


    # ------------------------------------
    # Post Training
    # ------------------------------------
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(work_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # Save latest model
    latest_path = os.path.join(work_dir, "latest_model.pth")
    torch.save(system.model.state_dict(), latest_path)

    # Save best model
    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()
    best_path = os.path.join(work_dir, "best_model.pth")
    torch.save(system.model.state_dict(), best_path)

    # ------------------------------------
    # Testing
    # ------------------------------------
    test_dir = testing_cfg.get("test_dir", None)
    if test_dir is not None:
        # Latest
        model.load_pretrain(latest_path)
        model.to(device)
        model.eval()
        latest_sisnr, latest_sdr, latest_stoi = val_librimix(model, model_cfg, dataset_cfg, testing_cfg, device)
        # Best
        model.load_pretrain(best_path)
        model.to(device)
        model.eval()
        best_sisnr, best_sdr, best_stoi = val_librimix(model, model_cfg, dataset_cfg, testing_cfg, device)
        # Save results
        log_path = os.path.join(work_dir, "log.txt")
        with open(log_path, "w") as outfile:
            outfile.write("best_sisnr" + ":" + str(best_sisnr) + '\n')
            outfile.write("best_sdr" + ":" + str(best_sdr) + '\n')
            outfile.write("best_stoi" + ":" + str(best_stoi) + '\n')
            outfile.write("latest_sisnr" + ":" + str(latest_sisnr) + '\n')
            outfile.write("latest_sdr" + ":" + str(latest_sdr) + '\n')
            outfile.write("latest_stoi" + ":" + str(latest_stoi) + '\n')
        if wandbLogger is not None:
            wandb.log({"latest_sisnr": latest_sisnr, "latest_sdr": latest_sdr, "latest_stoi": latest_stoi})
            wandb.log({"best_sisnr": best_sisnr, "best_sdr": best_sdr, "best_stoi": best_stoi})

