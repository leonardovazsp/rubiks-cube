import os
import torch
import wandb
import multiprocessing
import ctypes
from multiprocessing import Queue, Value, Process
from evaluator import Evaluator
from agent import Agent, Model
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
    weight_decay = 0.0001,
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

    model = Model(
                    device=device,
                    batch_size=batch_size,
                    lr=learning_rate,
                    warmup_steps=warmup_steps,
                    gamma=gamma,
                    checkpoint=checkpoint,
                    optimizer=optimizer,
                    weight_decay=weight_decay,
                )
    
    
    total_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if wandb_token:
        wandb_config = {"optimizer": optimizer,
                        "weight_decay": weight_decay,
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
    agent = Agent(memory, batch_size, reward, n_scrambles)
    model.set_memory(memory)
    stop_signal = Value(ctypes.c_bool, False)
    processes = start_processes(target=agent.generate_examples, num_processes=12, stop_signal=stop_signal)

    time.sleep(5)
    process = Process(target=evaluator.run)
    process.start()

    # def init_weights(layer):
    #     if type(layer) == torch.nn.Linear:
    #         torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    #         torch.nn.init.constant_(layer.bias, 0)
    #     elif type(layer) == torch.nn.Embedding:
    #         torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

    # print(f"Total model parameters: {total_model_params}. Initializing weights...")
    # agent.apply(init_weights)
    # print("Weights initialized!")

    # for p in agent.parameters():
    #     torch.nn.init.normal_(p, mean=0.0, std=0.05)
        # if p.dim() > 1:
        #     torch.nn.init.kaiming_normal_(p, nonlinearity='relu')
    
    for episode in range(episodes):
        if episode >= warmup_episodes and checkpoint:
            if os.path.exists(f'models/{checkpoint}'):
                model.load_state_dict(torch.load(f'models/{checkpoint}', map_location=device))

        pbar = tqdm(total=iterations)

        for iteration in range(iterations):
            loss = model.train_step()
            if wandb_token:
                wandb.log({"loss": loss})

            pbar.update(1)
            pbar.set_description(f"Episode {episode + 1} - iteration {iteration + 1} - loss: {loss:.4f}")

            if iteration % eval_every == 0 and iteration > 0:
                model.cpu()
                state_dict = model.state_dict().copy()
                evaluator.add_model((state_dict, episode + 1))
                model.to(device)

        pbar.close()
        

        if episode >= warmup_episodes:
            checkpoint = f'{run_name}_episode_{episode + 1}_best.pt'
        else:
            print(f"Warmup episode")
            checkpoint = f'{run_name}_latest.pt'

        print(f"Learning rate: {model.scheduler.get_last_lr()}")

    queue.put("STOP")
    process.join()

if __name__ == '__main__':
    import time
    multiprocessing.set_start_method('spawn', force=True)
    Fire(main)