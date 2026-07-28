"""Continue v2 training from latest checkpoint."""
import sys

sys.path.insert(0, '.')
from model.pico_type.train import TrainConfig, train

latest = 'checkpoints_v2/step_2500.pt'
cfg = TrainConfig(
    total_steps=6700,
    batch_size=16,
    train_size=10000,
    eval_size=500,
    lr=1e-3,
    log_every=50,
    save_every=500,
    eval_every=500,
    output_dir='checkpoints_v2',
    tier='base',
    train_tiers=('base',),
    device='mps',
    resume_from=latest,
    real_code_path='model/pico_type/data/real/code_samples.json',
    real_text_path='model/pico_type/data/real/text_samples.json',
    real_ratio=0.5,
)
print(f'Resuming v2 from {latest} to step {cfg.total_steps}...')
train(cfg)
