#!/usr/bin/env python
"""AlphaZero-style MCTS self-play with elo conditioning.

Architecture:
- Policy: model's move logits (softmax over legal moves)
- Value: separate linear probe on last hidden state (trained jointly)
- MCTS: UCB selection with policy prior + value backup
- Training: policy targets = MCTS visit distribution, value targets = game outcome

Self-play protocol:
- 50% same-elo games (both sides same elo, train both)
- 50% different-elo games (high vs low, only train high-elo side)
- MCTS with N simulations per move
- Record (token_prefix, mcts_policy, game_outcome) for each move

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/alphazero_selfplay.py \
        --hf-model ~/user_data/chess-v3/hf_export/600M_v3_step75K \
        --iterations 20 --games-per-iter 50 --mcts-sims 25
"""

import argparse
import json
import math
import os
import random
import sys
import time

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.data.tokenizer import Tokenizer as T
from src.data.tokens import CHESS_MOVES

MOVE_OFFSET = 378
tok2uci = {i + MOVE_OFFSET: m for i, m in enumerate(CHESS_MOVES)}
uci2tok = {v: k for k, v in tok2uci.items()}
STOCKFISH = "/home/yimingz3/bin/stockfish"


def make_prompt(elo_w, elo_b, tc=180, inc=0):
    # Use old vocab token IDs (model trained with vocab_size=2350)
    # Old vocab: elo=0-9, inc=10-191, sec=192-377, moves=378-2345, term=2346-2347, bos=2348
    p = [2348]  # old BOS
    # seconds token: index 192 + position in SECONDS_PER_SIDE list
    from src.data.tokens import SECONDS_PER_SIDE, INCREMENTS

    sec_str = str(tc)
    if sec_str in SECONDS_PER_SIDE:
        p.append(192 + SECONDS_PER_SIDE.index(sec_str))
    else:
        p.append(192 + len(SECONDS_PER_SIDE) - 1)  # wildcard
    inc_str = str(inc)
    if inc_str in INCREMENTS:
        p.append(10 + INCREMENTS.index(inc_str))
    else:
        p.append(10 + len(INCREMENTS) - 1)
    for d in f"{elo_w:04d}":
        p.append(int(d))
    for d in f"{elo_b:04d}":
        p.append(int(d))
    return p


def get_legal(board):
    return [uci2tok[m.uci()] for m in board.legal_moves if m.uci() in uci2tok]


class MCTSNode:
    __slots__ = ["move_tok", "parent", "children", "visits", "value_sum", "prior"]

    def __init__(self, move_tok=None, parent=None, prior=0.0):
        self.move_tok = move_tok
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def q_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def get_policy_and_value(model, token_prefix, legal_tokens, device):
    """Get policy (over legal moves) and value from the model."""
    input_ids = torch.tensor(token_prefix, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = input_ids.clamp(max=model.config.vocab_size - 1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    logits = outputs.logits[0, -1]  # [vocab]

    # Policy: softmax over legal moves
    legal_logits = {tok: logits[tok].item() for tok in legal_tokens}
    max_logit = max(legal_logits.values())
    exp_logits = {tok: math.exp(l - max_logit) for tok, l in legal_logits.items()}
    total = sum(exp_logits.values())
    policy = {tok: p / total for tok, p in exp_logits.items()}

    # Value: use model's hidden state + value head if available
    # For now, use a simple heuristic: value = 0 (will be trained)
    hidden = outputs.hidden_states[-1][0, -1]  # [hidden_size]
    if hasattr(model, "value_head"):
        value = torch.tanh(model.value_head(hidden.to(model.value_head.weight.dtype))).item()
    else:
        value = 0.0

    return policy, value


def mcts_search(model, board, token_prefix, device, n_sims=25, c_puct=1.5):
    """Run MCTS from current position, return visit distribution."""
    legal_tokens = get_legal(board)
    if not legal_tokens:
        return {}, 0.0

    # Root expansion
    policy, root_value = get_policy_and_value(model, token_prefix, legal_tokens, device)
    root = MCTSNode()
    for tok, prior in policy.items():
        root.children.append(MCTSNode(move_tok=tok, parent=root, prior=prior))
    root.visits = 1
    root.value_sum = root_value

    for _ in range(n_sims):
        node = root
        sim_board = board.copy()
        sim_tokens = list(token_prefix)

        # Selection: traverse tree using UCB
        while node.children and node.visits > 0:
            sign = 1 if sim_board.turn == chess.WHITE else -1
            best = max(
                node.children,
                key=lambda c: (
                    sign * c.q_value + c_puct * c.prior * math.sqrt(node.visits) / (1 + c.visits)
                ),
            )
            node = best
            if node.move_tok is not None:
                uci = tok2uci.get(node.move_tok)
                if uci:
                    try:
                        move = chess.Move.from_uci(uci)
                        if move in sim_board.legal_moves:
                            sim_board.push(move)
                            sim_tokens.append(node.move_tok)
                    except Exception:
                        break

        # Expansion + evaluation
        if not sim_board.is_game_over() and len(sim_tokens) < 1020:
            sim_legal = get_legal(sim_board)
            if sim_legal:
                policy, value = get_policy_and_value(model, sim_tokens, sim_legal, device)
                for tok, prior in policy.items():
                    node.children.append(MCTSNode(move_tok=tok, parent=node, prior=prior))
            else:
                value = 0.0
        else:
            if sim_board.is_game_over():
                outcome = sim_board.outcome()
                value = (
                    1.0
                    if outcome and outcome.winner == chess.WHITE
                    else (-1.0 if outcome and outcome.winner == chess.BLACK else 0.0)
                )
            else:
                value = 0.0

        # Backprop (value always from white's perspective)
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    # Return visit distribution
    visit_dist = {}
    total_visits = sum(c.visits for c in root.children)
    for c in root.children:
        if c.visits > 0:
            visit_dist[c.move_tok] = c.visits / total_visits

    return visit_dist, root_value


def play_game_with_mcts(model, elo_w, elo_b, device, n_sims=25):
    """Play a game using MCTS, return training data."""
    board = chess.Board()
    tokens = make_prompt(elo_w, elo_b)
    training_data = []  # [(token_prefix, mcts_policy_dict, None)]

    for move_num in range(200):
        if board.is_game_over():
            break

        legal = get_legal(board)
        if not legal:
            break

        # MCTS search
        visit_dist, _ = mcts_search(model, board, tokens, device, n_sims)
        if not visit_dist:
            break

        # Record training data
        training_data.append(
            {
                "prefix": list(tokens),
                "mcts_policy": visit_dist,
                "turn_is_white": board.turn == chess.WHITE,
            }
        )

        # Sample move from visit distribution (with temperature)
        temp = 1.0 if move_num < 15 else 0.1  # explore early, exploit later
        if temp < 0.5:
            # Greedy
            chosen_tok = max(visit_dist, key=visit_dist.get)
        else:
            # Sample proportional to visits^(1/temp)
            toks = list(visit_dist.keys())
            probs = [visit_dist[t] ** (1.0 / temp) for t in toks]
            total = sum(probs)
            probs = [p / total for p in probs]
            chosen_tok = random.choices(toks, probs)[0]

        uci = tok2uci.get(chosen_tok)
        if not uci:
            break
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break

        board.push(move)
        tokens.append(chosen_tok)

    # Game outcome
    result = board.result()
    value = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)

    # Assign values to training data
    for td in training_data:
        td["value"] = value if td["turn_is_white"] else -value

    return training_data, result, board.fullmove_number


def train_on_mcts_data(model, all_data, optimizer, device, epochs=1):
    """Train policy to match MCTS distribution, value to predict outcome."""
    model.train()
    total_policy_loss = 0
    total_value_loss = 0
    n_updates = 0

    for epoch in range(epochs):
        random.shuffle(all_data)
        for td in all_data:
            prefix = td["prefix"]
            mcts_policy = td["mcts_policy"]
            target_value = td["value"]

            input_ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
            input_ids = input_ids.clamp(max=model.config.vocab_size - 1)

            outputs = model(input_ids=input_ids, output_hidden_states=True)
            logits = outputs.logits[0, -1]  # [vocab]

            # Policy loss: KL divergence from MCTS distribution
            legal_toks = list(mcts_policy.keys())
            target_probs = torch.tensor([mcts_policy[t] for t in legal_toks], device=device)
            pred_logits = torch.stack([logits[t] for t in legal_toks])
            pred_log_probs = F.log_softmax(pred_logits, dim=-1)
            policy_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

            # Value loss (if value head exists)
            value_loss = torch.tensor(0.0, device=device)
            if hasattr(model, "value_head"):
                hidden = outputs.hidden_states[-1][0, -1]
                pred_value = torch.tanh(model.value_head(hidden.to(model.value_head.weight.dtype)))
                target_v = torch.tensor(target_value, device=device, dtype=pred_value.dtype)
                value_loss = F.mse_loss(pred_value.squeeze(), target_v)

            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_updates += 1

    model.eval()
    return (total_policy_loss / max(n_updates, 1), total_value_loss / max(n_updates, 1))


def quick_eval_greedy(model, device, n_games=10):
    """Quick eval using greedy (no MCTS) against SF."""
    results = {}
    for sf_level in [1, 3, 5]:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        engine.configure({"Skill Level": sf_level, "Threads": 1})
        w = d = l = 0

        for i in range(n_games):
            board = chess.Board()
            tokens = make_prompt(2000, 2000)
            mw = i % 2 == 0

            for _ in range(400):
                if board.is_game_over():
                    break
                if (board.turn == chess.WHITE) == mw:
                    legal = get_legal(board)
                    if not legal:
                        break
                    # Greedy from model
                    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                    input_ids = input_ids.clamp(max=model.config.vocab_size - 1)
                    with torch.no_grad():
                        logits = model(input_ids=input_ids).logits[0, -1]
                    legal_logits = [(t, logits[t].item()) for t in legal]
                    tok = max(legal_logits, key=lambda x: x[1])[0]
                    uci = tok2uci.get(tok)
                    if not uci or chess.Move.from_uci(uci) not in board.legal_moves:
                        break
                    board.push(chess.Move.from_uci(uci))
                    tokens.append(tok)
                else:
                    sf = engine.play(board, chess.engine.Limit(depth=sf_level))
                    board.push(sf.move)
                    uci = sf.move.uci()
                    if uci in uci2tok:
                        tokens.append(uci2tok[uci])

            r = board.result()
            if (r == "1-0" and mw) or (r == "0-1" and not mw):
                w += 1
            elif r in ("1/2-1/2", "*"):
                d += 1
            else:
                l += 1

        engine.quit()
        results[sf_level] = (w, d, l)
        print(f"  SF{sf_level}: {w}W {d}D {l}L")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--games-per-iter", type=int, default=20)
    parser.add_argument("--mcts-sims", type=int, default=25)
    parser.add_argument("--elo-high", type=int, default=2500)
    parser.add_argument("--elo-low", type=int, default=1800)
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="results/alphazero")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"
    log = []

    print(
        f"AlphaZero self-play: {args.iterations} iters, "
        f"{args.games_per_iter} games/iter, {args.mcts_sims} sims/move"
    )

    # Load model
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    # Add value head
    hidden_size = model.config.hidden_size
    model.value_head = torch.nn.Linear(hidden_size, 1).to(device).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Initial eval
    print("\n=== Baseline ===")
    replay_buffer = []  # accumulate training data across iterations
    eval0 = quick_eval_greedy(model, device, args.eval_games)
    log.append({"iter": 0, "eval": {k: list(v) for k, v in eval0.items()}})

    for it in range(1, args.iterations + 1):
        print(f"\n{'=' * 50}")
        print(f"Iteration {it}/{args.iterations}")

        all_data = []
        t0 = time.time()

        for g in range(args.games_per_iter):
            # 50% same elo, 50% different elo
            if random.random() < 0.5:
                elo_w = elo_b = random.choice([1800, 2000, 2200, 2400])
                train_both = True
            else:
                elo_w, elo_b = args.elo_high, args.elo_low
                if random.random() < 0.5:
                    elo_w, elo_b = elo_b, elo_w  # swap sides
                train_both = False

            data, result, n_moves = play_game_with_mcts(model, elo_w, elo_b, device, args.mcts_sims)

            # Filter: for different-elo games, only keep high-elo side's data
            if not train_both:
                high_is_white = elo_w == args.elo_high
                data = [d for d in data if d["turn_is_white"] == high_is_white]

            all_data.extend(data)

            if (g + 1) % 5 == 0:
                print(
                    f"  Game {g + 1}/{args.games_per_iter}: {result} ({n_moves}mv), "
                    f"{len(all_data)} training positions"
                )

        gen_time = time.time() - t0
        print(f"  Generated in {gen_time:.0f}s, {len(all_data)} training positions")

        # Train
        print(f"\n[TRAIN]")
        p_loss, v_loss = train_on_mcts_data(model, all_data, optimizer, device, epochs=1)
        print(f"  Policy loss: {p_loss:.4f}, Value loss: {v_loss:.4f}")

        # Eval (greedy, no MCTS)
        print(f"\n[EVAL]")
        eval_r = quick_eval_greedy(model, device, args.eval_games)

        log.append(
            {
                "iter": it,
                "eval": {k: list(v) for k, v in eval_r.items()},
                "n_positions": len(all_data),
                "policy_loss": p_loss,
                "value_loss": v_loss,
                "gen_time": gen_time,
            }
        )

        with open(os.path.join(args.output_dir, "log.json"), "w") as f:
            json.dump(log, f, indent=2)

    # Save final model
    if hasattr(model, "value_head"):
        vh = model.value_head
        delattr(model, "value_head")
        model.save_pretrained(os.path.join(args.output_dir, "model"))
        torch.save(vh.state_dict(), os.path.join(args.output_dir, "value_head.pt"))
    else:
        model.save_pretrained(os.path.join(args.output_dir, "model"))

    print("\nDone!")


if __name__ == "__main__":
    main()


def train_mixed(model, mcts_data, optimizer, device, pretrain_data_path, mix_ratio=5, batch_size=8, seq_len=1024):
    """Train on MCTS data + pretraining data mix to prevent forgetting.
    
    For every MCTS position, also train on mix_ratio pretraining sequences.
    """
    import numpy as np
    
    model.train()
    total_mcts_loss = 0
    total_pretrain_loss = 0
    n_mcts = 0
    n_pretrain = 0
    
    # Load pretraining data (memmap for efficiency)
    pretrain = np.memmap(pretrain_data_path, dtype=np.uint16, mode='r')
    pretrain_len = len(pretrain) // seq_len
    
    random.shuffle(mcts_data)
    
    for td in mcts_data:
        # 1. MCTS training step
        prefix = td["prefix"]
        mcts_policy = td["mcts_policy"]
        target_value = td["value"]
        
        input_ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
        input_ids = input_ids.clamp(max=model.config.vocab_size - 1)
        
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits[0, -1]
        
        legal_toks = list(mcts_policy.keys())
        target_probs = torch.tensor([mcts_policy[t] for t in legal_toks], device=device)
        pred_logits = torch.stack([logits[t] for t in legal_toks])
        pred_log_probs = F.log_softmax(pred_logits, dim=-1)
        policy_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")
        
        value_loss = torch.tensor(0.0, device=device)
        if hasattr(model, "value_head"):
            hidden = outputs.hidden_states[-1][0, -1]
            pred_value = torch.tanh(model.value_head(hidden.to(model.value_head.weight.dtype)))
            target_v = torch.tensor(target_value, device=device, dtype=pred_value.dtype)
            value_loss = F.mse_loss(pred_value.squeeze(), target_v)
        
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_mcts_loss += loss.item()
        n_mcts += 1
        
        # 2. Pretraining data steps (mix_ratio steps per MCTS step)
        for _ in range(mix_ratio):
            idx = random.randint(1, pretrain_len - 2)  # skip first (corrupted)
            seq = pretrain[idx * seq_len : (idx + 1) * seq_len].astype(np.int64)
            seq = torch.tensor(seq, device=device).unsqueeze(0)
            seq = seq.clamp(max=model.config.vocab_size - 1)
            
            pt_outputs = model(input_ids=seq[:, :-1])
            pt_logits = pt_outputs.logits
            pt_targets = seq[:, 1:]
            
            # Only compute loss on move tokens (378-2345)
            move_mask = (pt_targets >= 378) & (pt_targets <= 2345)
            if move_mask.any():
                pt_loss = F.cross_entropy(
                    pt_logits[move_mask], pt_targets[move_mask]
                )
                optimizer.zero_grad()
                pt_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_pretrain_loss += pt_loss.item()
                n_pretrain += 1
    
    model.eval()
    avg_mcts = total_mcts_loss / max(n_mcts, 1)
    avg_pt = total_pretrain_loss / max(n_pretrain, 1)
    return avg_mcts, avg_pt
