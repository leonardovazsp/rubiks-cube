import os
import torch
import wandb
import multiprocessing
import ctypes
from multiprocessing import Queue, Value, Process
from evaluator import Evaluator
from agent import Agent
from tqdm import tqdm
from fire import Fire

if not os.path.exists("models"):
    os.makedirs("models")

def verbose_print(text, verbose):
    if verbose:
        print(text)

def start_processes(target, num_processes, stop_signal):
    processes = []
    for _ in range(num_processes):
        process = Process(target=target, args=(stop_signal,))
        process.start()
        processes.append(process)
    return processes

def main(
    batch_size = 512,
    n_scrambles = 8,
    episodes = 1000,
    iterations = 1000,
    eval_every = 100,
    learning_rate = 0.0001,
    gamma = 0.9999,
    optimizer = "Adam",
    device = "cuda:0",
    eval_device = "cuda:1",
    checkpoint = None,
    warmup_steps = 100,
    reward = 1.0,
    verbose = False,
    wandb_token = None,
    warmup_episodes = 10,
    project = "rubiks-cube-reinforcement-learning"
):
    
    evaluator = Evaluator(
        test_cubes_path='test_cubes.pkl',
        device=eval_device
    )

    agent = Agent(
                    device=device,
                    batch_size=batch_size,
                    lr=learning_rate,
                    warmup_steps=warmup_steps,
                    gamma=gamma,
                    checkpoint=checkpoint,
                    reward=reward,
                    optimizer=optimizer,
                    scrambles=n_scrambles
                )
    
    total_model_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    if wandb_token:
        wandb_config = {"optimizer": optimizer,
                        "batch_size": batch_size,
                        "n_scrambles": n_scrambles,
                        "lr_decay": gamma,
                        "learning_rate": learning_rate,
                        "params": total_model_params,
                        "reward": reward,
                        "checkpoint": checkpoint}
        
        wandb.init(project=project, config=wandb_config)
        run_id = wandb.run.id
        run_name = wandb.run.name
        
        evaluator.init_wandb(run_id, project)

    else:
        run_name = "model"
    
    queue = Queue()
    evaluator.set_queue(queue)
    

    memory = Queue()
    agent.set_memory(memory)
    stop_signal = Value(ctypes.c_bool, False)
    processes = start_processes(target=agent.generate_examples, num_processes=12, stop_signal=stop_signal)

    for layer in agent.parameters():
        torch.nn.init.normal_(layer, mean=0.0, std=0.05)
    
    for episode in range(episodes):
        process = Process(target=evaluator.run)
        process.start()

        if episode >= warmup_episodes and checkpoint:
            if os.path.exists(f'models/{checkpoint}'):
                agent.load_state_dict(torch.load(f'models/{checkpoint}', map_location=device))

        pbar = tqdm(total=iterations)

        for iteration in range(iterations):
            loss = agent.train_step()
            if wandb_token:
                wandb.log({"loss": loss})

            pbar.update(1)
            pbar.set_description(f"Episode {episode + 1} - iteration {iteration + 1} - loss: {loss:.4f}")

            if iteration % eval_every == 0 and iteration > 0:
                agent.cpu()
                state_dict = agent.state_dict().copy()
                evaluator.add_model((state_dict, episode + 1))
                agent.to(device)

        pbar.close()
        queue.put("STOP")
        process.join()

        if episode >= warmup_episodes:
            checkpoint = f'{run_name}_episode_{episode + 1}_best.pt'
        else:
            print(f"Warmup episode {episode + 1} - loss: {loss:.4f}")
            checkpoint = f'{run_name}_latest.pt'

if __name__ == '__main__':
    import time
    multiprocessing.set_start_method('spawn', force=True)
    Fire(main)