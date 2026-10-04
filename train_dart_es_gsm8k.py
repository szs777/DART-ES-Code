import argparse
from datetime import datetime
import json
import math
import os
import random
import signal
import sys

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port


from gsm8k.reward_function import reward_function


SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 4
EXPERIMENT_DIR = "outputs/dart-es-gsm8k-qwen2.5-1.5b"
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gsm8k",
    "gsm8k_main_train.json",
)



EPOCHS = 40
CHUNK_SIZE = 200
REPLAY_RATIO = 0.20
PASS_EMA_DECAY = 0.70
WEIGHT_FLOOR = 0.80
WEIGHT_EXPONENT = 3.0
REPLAY_PASS_THRESHOLD = 0.50
MAX_REPLAYS_PER_EPOCH = 2
REPLAY_COOLDOWN_ITERATIONS = 1
ZERO_PASS_PATIENCE = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="DART-ES fine-tuning for GSM8K with multi-engine NCCL synchronization"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the processed GSM8K training JSON.",
    )
    parser.add_argument("--replay_ratio", type=float, default=REPLAY_RATIO)
    parser.add_argument("--pass_ema_decay", type=float, default=PASS_EMA_DECAY)
    parser.add_argument("--weight_floor", type=float, default=WEIGHT_FLOOR)
    parser.add_argument("--weight_exponent", type=float, default=WEIGHT_EXPONENT)
    parser.add_argument("--replay_pass_threshold", type=float, default=REPLAY_PASS_THRESHOLD)
    parser.add_argument("--max_replays_per_epoch", type=int, default=MAX_REPLAYS_PER_EPOCH)
    parser.add_argument(
        "--replay_cooldown_iterations",
        type=int,
        default=REPLAY_COOLDOWN_ITERATIONS,
    )
    parser.add_argument("--zero_pass_patience", type=int, default=ZERO_PASS_PATIENCE)
    parser.add_argument(
        "--global_seed",
        type=int,
        help="Global random seed",
    )
    args = parser.parse_args()

    if args.population_size <= 0:
        parser.error("--population_size must be positive")
    if args.population_size > 1_000_001:
        parser.error("--population_size cannot exceed 1,000,001 with unique seed sampling")
    if args.chunk_size <= 0:
        parser.error("--chunk_size must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if not 0.0 <= args.replay_ratio < 1.0:
        parser.error("--replay_ratio must be in [0, 1)")
    if not 0.0 <= args.pass_ema_decay < 1.0:
        parser.error("--pass_ema_decay must be in [0, 1)")
    if args.weight_floor < 0.0:
        parser.error("--weight_floor must be non-negative")
    if args.weight_exponent < 0.0:
        parser.error("--weight_exponent must be non-negative")
    if not 0.0 < args.replay_pass_threshold <= 1.0:
        parser.error("--replay_pass_threshold must be in (0, 1]")
    if args.max_replays_per_epoch < 0:
        parser.error("--max_replays_per_epoch must be non-negative")
    if args.replay_cooldown_iterations < 0:
        parser.error("--replay_cooldown_iterations must be non-negative")
    if args.zero_pass_patience <= 0:
        parser.error("--zero_pass_patience must be positive")

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    
    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)

    return args


def sample_key(task_data):
    
    if "id" not in task_data:
        raise KeyError("Every training sample must contain a stable 'id' field")
    return str(task_data["id"])


def difficulty_weight(pass_rate_ema, floor=WEIGHT_FLOOR, exponent=WEIGHT_EXPONENT):
    
    return float(floor + math.exp(-float(exponent) * float(pass_rate_ema)))


def replay_priority(pass_rate_ema, exponent=WEIGHT_EXPONENT):
    
    return float(math.exp(-float(exponent) * float(pass_rate_ema)))


def format_distribution(name, values):
    values = np.asarray(values, dtype=np.float64)
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return (
        f"{name}: mean={float(np.mean(values)):.6f}, "
        f"std={float(np.std(values)):.6f}, "
        f"p10={float(p10):.6f}, p50={float(p50):.6f}, p90={float(p90):.6f}"
    )


def weighted_sample_without_replacement(items, weights, count):
    
    if count <= 0 or not items:
        return []
    count = min(int(count), len(items))
    probabilities = np.asarray(weights, dtype=np.float64)
    probability_sum = float(np.sum(probabilities))
    if not np.isfinite(probability_sum) or probability_sum <= 0.0:
        probabilities = None
    else:
        probabilities = probabilities / probability_sum
    selected_positions = np.random.choice(
        len(items), size=count, replace=False, p=probabilities
    )
    return [items[int(position)] for position in np.atleast_1d(selected_positions)]


class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def launch_engines(num_engines, model_name):
    
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype="float16",
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for strategy in strategies
    ]
    return engines, pgs


def evaluate_gsm8k_handle(llm, task_datas):
    
    prompts = [d["context"] for d in task_datas]
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=1024,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle



def _postprocess_outputs(outputs, task_datas):
    answer_rewards = []
    format_rewards = []
    for output, data in zip(outputs, task_datas):
        response = output.outputs[0].text

        r = reward_function(
            response=response,
            numbers=None,
            target=data["answer"],
            end_token=None,
        )
        answer_rewards.append(float(r["reward_info"]["answer_reward"]))
        format_rewards.append(float(r["reward_info"]["format_reward"]))
    return {
        "answer_rewards": answer_rewards,
        "format_rewards": format_rewards,
    }

def main(args):
    
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    
    run_dir = f"{args.experiment_dir}/dart_es_gsm8k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_saves_dir = f"{run_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    
    data_path = args.data_path
    print(f"Loading GSM8K data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        task_datas = json.load(f)
    num_samples = len(task_datas)
    print(f"Loaded {num_samples} GSM8K training samples")

    
    chunk_size = args.chunk_size
    replay_slots = int(round(chunk_size * args.replay_ratio))
    replay_slots = min(max(replay_slots, 0), chunk_size - 1)
    nominal_fresh_slots = chunk_size - replay_slots
    print(
        f"Will use batch size = {chunk_size}: nominally "
        f"{nominal_fresh_slots} fresh + {replay_slots} replay samples."
    )

    data_by_key = {}
    data_state = {}
    for data in task_datas:
        key = sample_key(data)
        if key in data_by_key:
            raise ValueError(f"Duplicate GSM8K sample id: {key}")
        data_by_key[key] = data
        data_state[key] = {
            "pass_rate_ema": None,
            "replay_count_epoch": 0,
            "last_replay_iteration": -1_000_000_000,
            "consecutive_zero_pass_count": 0,
        }
    replay_buffer = set()

    
    engines, pgs = launch_engines(args.num_engines, args.model_name)

    
    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])

    def cleanup():
        for llm in engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()

    def sig_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    def eligible_replay_ids(excluded_ids, iteration_idx):
        
        candidates = []
        priorities = []
        for key in sorted(replay_buffer):
            if key in excluded_ids:
                continue
            state = data_state[key]
            if state["replay_count_epoch"] >= args.max_replays_per_epoch:
                continue
            if (
                iteration_idx - state["last_replay_iteration"]
                <= args.replay_cooldown_iterations
            ):
                continue
            pass_rate_ema = state["pass_rate_ema"]
            if pass_rate_ema is None:
                continue
            if pass_rate_ema >= args.replay_pass_threshold:
                continue
            candidates.append(key)
            priorities.append(replay_priority(pass_rate_ema, args.weight_exponent))
        return candidates, priorities

    def draw_replay_ids(count, excluded_ids, iteration_idx):
        
        candidates, priorities = eligible_replay_ids(excluded_ids, iteration_idx)
        selected = weighted_sample_without_replacement(candidates, priorities, count)
        for key in selected:
            state = data_state[key]
            state["replay_count_epoch"] += 1
            state["last_replay_iteration"] = iteration_idx
        return selected

    
    def evaluate_population(seeds, task_datas_for_iter):
        
        seeds_perf = {}
        seed_iter = iter(seeds)
        inflight = {}

        
        for llm in engines:
            try:
                seed = next(seed_iter)
            except StopIteration:
                break

            
            ray.get(llm.collective_rpc.remote(
                "perturb_self_weights",
                args=(seed, args.sigma, False)
            ))
            handle = evaluate_gsm8k_handle(llm, task_datas_for_iter)
            inflight[handle] = {
                "engine": llm,
                "seed": seed,
            }

        
        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            metrics = _postprocess_outputs(outputs, task_datas_for_iter)

            seeds_perf[meta["seed"]] = metrics

            llm = meta["engine"]
            
            ray.get(llm.collective_rpc.remote(
                "restore_self_weights",
                args=(meta["seed"], args.sigma)
            ))

            
            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(llm.collective_rpc.remote(
                "perturb_self_weights",
                args=(next_seed, args.sigma, False)
            ))
            handle = evaluate_gsm8k_handle(llm, task_datas_for_iter)
            inflight[handle] = {
                "engine": llm,
                "seed": next_seed,
            }

        missing_seeds = [seed for seed in seeds if seed not in seeds_perf]
        if missing_seeds:
            raise RuntimeError(f"Population evaluation missed seeds: {missing_seeds}")

        ordered_seeds = list(seeds)
        batch_size = len(task_datas_for_iter)

        answer_matrix = np.asarray(
            [seeds_perf[seed]["answer_rewards"] for seed in ordered_seeds],
            dtype=np.float64,
        )
        format_matrix = np.asarray(
            [seeds_perf[seed]["format_rewards"] for seed in ordered_seeds],
            dtype=np.float64,
        )
        expected_shape = (len(ordered_seeds), batch_size)
        if answer_matrix.shape != expected_shape or format_matrix.shape != expected_shape:
            raise RuntimeError(
                "Population reward matrix shape mismatch: "
                f"answer={answer_matrix.shape}, format={format_matrix.shape}, "
                f"expected={expected_shape}"
            )

        current_pass_rates = np.mean(answer_matrix, axis=0)
        correct_counts = np.sum(answer_matrix, axis=0).astype(np.int64)
        ema_pass_rates = np.zeros(batch_size, dtype=np.float64)
        sample_weights = np.zeros(batch_size, dtype=np.float64)

        for sample_idx, task_data in enumerate(task_datas_for_iter):
            key = sample_key(task_data)
            state = data_state[key]
            previous_ema = state["pass_rate_ema"]
            current_pass_rate = float(current_pass_rates[sample_idx])
            if previous_ema is None:
                updated_ema = current_pass_rate
            else:
                updated_ema = (
                    args.pass_ema_decay * float(previous_ema)
                    + (1.0 - args.pass_ema_decay) * current_pass_rate
                )

            correct_count = int(correct_counts[sample_idx])
            if correct_count == 0:
                state["consecutive_zero_pass_count"] += 1
            else:
                state["consecutive_zero_pass_count"] = 0

            if updated_ema >= args.replay_pass_threshold:
                replay_buffer.discard(key)
            elif (
                correct_count == 0
                and state["consecutive_zero_pass_count"] >= args.zero_pass_patience
            ):
                replay_buffer.discard(key)
            elif correct_count > 0:
                replay_buffer.add(key)

            state["pass_rate_ema"] = float(updated_ema)

            weight = difficulty_weight(
                updated_ema,
                floor=args.weight_floor,
                exponent=args.weight_exponent,
            )
            ema_pass_rates[sample_idx] = updated_ema
            sample_weights[sample_idx] = weight

        weighted_answer_scores = np.mean(
            answer_matrix * sample_weights[np.newaxis, :], axis=1
        )
        raw_format_scores = np.mean(format_matrix, axis=1)
        weighted_fitnesses = weighted_answer_scores + 0.1 * raw_format_scores
        raw_population_rewards = np.mean(
            answer_matrix + 0.1 * format_matrix,
            axis=1,
        )

        print(format_distribution("population_reward", raw_population_rewards))
        print(format_distribution("weighted_fitness", weighted_fitnesses))
        print(format_distribution("current_pass_rate", current_pass_rates))
        print(format_distribution("ema_pass_rate", ema_pass_rates))
        print(format_distribution("difficulty_weight", sample_weights))
        print(f"replay_pool_size={len(replay_buffer)}")

        weighted_mean = float(np.mean(weighted_fitnesses))
        weighted_std = float(np.std(weighted_fitnesses))
        for seed_idx, seed in enumerate(ordered_seeds):
            normalized_fitness = (
                float(weighted_fitnesses[seed_idx]) - weighted_mean
            ) / (weighted_std + 1e-8)
            seeds_perf[seed]["norm_reward"] = normalized_fitness
        return seeds_perf

    
    epochs = args.epochs
    global_iter = 0  

    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")

        for state in data_state.values():
            state["replay_count_epoch"] = 0

        
        indices = list(range(num_samples))
        random.shuffle(indices)
        fresh_cursor = 0
        pos_in_epoch = 0
        approximate_chunks = math.ceil(num_samples / max(nominal_fresh_slots, 1))
        print(
            f"Epoch {epoch + 1}: full fresh-data coverage; approximately "
            f"{approximate_chunks} batches once replay slots are filled."
        )

        while fresh_cursor < num_samples:
            replay_pool_size_before = len(replay_buffer)

            
            
            
            fresh_window_indices = indices[
                fresh_cursor:min(fresh_cursor + chunk_size, num_samples)
            ]
            fresh_window_ids = {
                sample_key(task_datas[data_idx]) for data_idx in fresh_window_indices
            }
            selected_replay_ids = draw_replay_ids(
                replay_slots,
                excluded_ids=fresh_window_ids,
                iteration_idx=global_iter,
            )

            fresh_capacity = chunk_size - len(selected_replay_ids)
            selected_fresh_indices = indices[
                fresh_cursor:min(fresh_cursor + fresh_capacity, num_samples)
            ]
            fresh_cursor += len(selected_fresh_indices)

            current_tasks = [
                task_datas[data_idx] for data_idx in selected_fresh_indices
            ]
            current_tasks.extend(data_by_key[key] for key in selected_replay_ids)
            random.shuffle(current_tasks)
            fresh_count = len(selected_fresh_indices)
            replay_count = len(selected_replay_ids)

            print(
                f"\n\n=== Generation {global_iter} "
                f"(epoch {epoch + 1}, batch {pos_in_epoch + 1}, "
                f"fresh={fresh_count}, replay={replay_count}, "
                f"pool={replay_pool_size_before}) ==="
            )

            base_seeds = random.sample(
                range(0, 1_000_001), k=args.population_size
            )
            base_seeds_perf = evaluate_population(base_seeds, current_tasks)

            
            final_per_seed_coeffs = [
                (seed, (args.alpha / args.population_size) * float(base_seeds_perf[seed]["norm_reward"]))
                for seed in base_seeds
            ]

            
            handles = []
            for seed, coeff in final_per_seed_coeffs:
                handles.append(
                    engines[0].collective_rpc.remote(
                        "perturb_self_weights",
                        args=(seed, coeff, False)
                    )
                )
            ray.get(handles)

            
            ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
            print(f"=== Generation {global_iter} finished ===\n")

            global_iter += 1  
            pos_in_epoch += 1

        
        epoch_model_path = f"{model_saves_dir}/final_model_epoch_{epoch + 1}"
        os.makedirs(epoch_model_path, exist_ok=True)
        
        epoch_pth_path = f"{epoch_model_path}/pytorch_model.pth"
        ray.get(
            engines[0].collective_rpc.remote(
                "save_self_weights_to_disk", args=(epoch_pth_path,)
            )
        )
        print(
            f"Model after epoch {epoch + 1} saved in .pth format to "
            f"{epoch_pth_path}."
        )

    print(f"Training finished. Models have been saved after each of {epochs} epochs.")

    cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
