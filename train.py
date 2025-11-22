import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from src.smirk_trainer import SmirkTrainer
import os
from datasets.data_utils import load_dataloaders
import copy


def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    # Remove the configuration file name from sys.argv for downstream CLI merging
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    return conf


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def try_create_base_encoder(trainer: SmirkTrainer):
    """
    Ensure trainer has a base_encoder usable for visualizations.
    Prefer trainer.create_base_encoder() if available; otherwise copy smirk_encoder.
    """
    if hasattr(trainer, "create_base_encoder") and callable(getattr(trainer, "create_base_encoder")):
        trainer.create_base_encoder()
    else:
        if not hasattr(trainer, "smirk_encoder"):
            raise AttributeError("trainer.smirk_encoder is missing; cannot create base encoder.")
        trainer.base_encoder = copy.deepcopy(trainer.smirk_encoder)
        trainer.base_encoder.eval()


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()

    # ----------------------- initialize log directories --------------------- #
    ensure_dir(config.train.log_path)
    train_images_save_path = os.path.join(config.train.log_path, 'train_images')
    val_images_save_path = os.path.join(config.train.log_path, 'val_images')
    ensure_dir(train_images_save_path)
    ensure_dir(val_images_save_path)
    OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    # ----------------------- dataloaders ------------------------------------ #
    train_loader, val_loader = load_dataloaders(config)

    # ----------------------- trainer ---------------------------------------- #
    trainer = SmirkTrainer(config)
    trainer = trainer.to(config.device)

    # resume checkpoint if requested
    if getattr(config, "resume", False):
        trainer.load_model(
            config.resume,
            load_fuse_generator=getattr(config, "load_fuse_generator", True),
            load_encoder=getattr(config, "load_encoder", True),
            device=config.device
        )

    # after loading, copy the base encoder (needed by visualizations)
    try_create_base_encoder(trainer)

    # training epochs
    start_epoch = getattr(config.train, "resume_epoch", 0)
    num_epochs = getattr(config.train, "num_epochs", getattr(config.train, "epochs", 1))
    visualize_every = getattr(config.train, "visualize_every", 10)
    save_every = getattr(config.train, "save_every", 1)

    for epoch in range(start_epoch, num_epochs):
        # restart everything at each epoch (schedulers T_max depend on steps)
        trainer.configure_optimizers(len(train_loader))

        for phase in ['train', 'val']:
            loader = train_loader if phase == 'train' else val_loader

            # ensure phase image dir exists
            phase_dir = train_images_save_path if phase == 'train' else val_images_save_path
            ensure_dir(phase_dir)

            for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
                if batch is None:
                    continue

                # freeze schedule
                trainer.set_freeze_status(config, batch_idx, epoch)

                # move tensors to device; keep non-tensors intact
                for key in batch:
                    val = batch[key]
                    if isinstance(val, torch.Tensor):
                        batch[key] = val.to(config.device)

                # single step
                outputs = trainer.step(batch, batch_idx, phase=phase)

                # visualize periodically
                if batch_idx % visualize_every == 0:
                    if outputs is not None and 'rendered_img' in outputs:
                        with torch.no_grad():
                            visualizations = trainer.create_visualizations(batch, outputs)
                            save_path = os.path.join(phase_dir, f"{epoch}_{batch_idx}.jpg")
                            trainer.save_visualizations(visualizations, save_path, show_landmarks=True)

        # save model periodically
        if epoch % save_every == 0:
            model_path = os.path.join(config.train.log_path, f"model_{epoch}.pt")
            trainer.save_model(trainer.state_dict(), model_path)
