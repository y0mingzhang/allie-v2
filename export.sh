

source .venv/bin/activate && python src/tools/hf/export_checkpoint.py --config configs/main_runs_v2/qwen-3-1.7b-57b.json --checkpoint 'models/main_runs_v2/qwen-3-1.7b-57b-cool-from-66550/96800/weights_tp_rank_world_size=0_1_pp_rank_world_size=0_1.pth' --output-dir export/qwen-3-1.7b-57b-cool-from-66550 --dtype bfloat16 --upload --repo-id yimingzhang/qwen-3-1.7b-57b-cool-from-66550-step96800
