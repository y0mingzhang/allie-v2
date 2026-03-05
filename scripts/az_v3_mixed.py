#!/usr/bin/env python
"""AlphaZero v3: MCTS self-play with pretraining data mixing.

Same as alphazero_selfplay.py but uses train_mixed() instead of train_on_mcts_data()
to prevent catastrophic forgetting by interleaving pretraining data.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/az_v3_mixed.py \
        --hf-model ~/user_data/chess-v3/hf_export/200M_v3_selfplay \
        --iterations 10 --games-per-iter 20 --mcts-sims 25 \
        --mix-ratio 5 --lr 5e-6
"""

import sys

sys.path.insert(0, ".")

# Import everything from the existing script
from scripts.alphazero_selfplay import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iter", type=int, default=20)
    parser.add_argument("--mcts-sims", type=int, default=25)
    parser.add_argument("--elo-high", type=int, default=2500)
    parser.add_argument("--elo-low", type=int, default=1800)
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--mix-ratio", type=int, default=5, help="Pretraining steps per MCTS step")
    parser.add_argument("--pretrain-data", default="data/tokens_v2/full_v3/train.npy")
    parser.add_argument("--output-dir", default="results/alphazero_200M_v3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"
    log = []

    print(
        f"AlphaZero v3 (mixed): {args.iterations} iters, {args.games_per_iter} games/iter, "
        f"{args.mcts_sims} sims, mix_ratio={args.mix_ratio}"
    )

    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    hidden_size = model.config.hidden_size
    model.value_head = torch.nn.Linear(hidden_size, 1).to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Baseline eval
    print("\n=== Baseline ===")
    eval0 = quick_eval_greedy(model, device, args.eval_games)
    log.append({"iter": 0, "eval": {k: list(v) for k, v in eval0.items()}})

    for it in range(1, args.iterations + 1):
        print(f"\n{'=' * 50}")
        print(f"Iteration {it}/{args.iterations}")

        all_data = []
        t0 = time.time()

        for g in range(args.games_per_iter):
            if random.random() < 0.5:
                elo_w = elo_b = random.choice([1800, 2000, 2200, 2400])
                train_both = True
            else:
                elo_w, elo_b = args.elo_high, args.elo_low
                if random.random() < 0.5:
                    elo_w, elo_b = elo_b, elo_w
                train_both = False

            data, result, n_moves = play_game_with_mcts(model, elo_w, elo_b, device, args.mcts_sims)

            if not train_both:
                high_is_white = elo_w == args.elo_high
                data = [d for d in data if d["turn_is_white"] == high_is_white]

            all_data.extend(data)

            if (g + 1) % 5 == 0:
                print(
                    f"  Game {g + 1}/{args.games_per_iter}: {result} ({n_moves}mv), "
                    f"{len(all_data)} positions"
                )

        gen_time = time.time() - t0
        print(f"  Generated in {gen_time:.0f}s, {len(all_data)} training positions")

        # Train with pretraining data mixing
        print(f"\n[TRAIN] MCTS + {args.mix_ratio}x pretraining mix")
        mcts_loss, pt_loss = train_mixed(
            model,
            all_data,
            optimizer,
            device,
            pretrain_data_path=args.pretrain_data,
            mix_ratio=args.mix_ratio,
        )
        print(f"  MCTS loss: {mcts_loss:.4f}, Pretrain loss: {pt_loss:.4f}")

        # Eval
        print(f"\n[EVAL]")
        eval_r = quick_eval_greedy(model, device, args.eval_games)

        log.append(
            {
                "iter": it,
                "eval": {k: list(v) for k, v in eval_r.items()},
                "n_positions": len(all_data),
                "mcts_loss": mcts_loss,
                "pretrain_loss": pt_loss,
                "gen_time": gen_time,
            }
        )

        with open(os.path.join(args.output_dir, "log.json"), "w") as f:
            json.dump(log, f, indent=2)

    # Save
    if hasattr(model, "value_head"):
        vh = model.value_head
        delattr(model, "value_head")
        model.save_pretrained(os.path.join(args.output_dir, "model"))
        torch.save(vh.state_dict(), os.path.join(args.output_dir, "value_head.pt"))

    print("\nDone!")


if __name__ == "__main__":
    main()
