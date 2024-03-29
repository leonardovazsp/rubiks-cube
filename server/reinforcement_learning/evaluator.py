import os
import json
import copy
import torch
import wandb
import pickle
import numpy as np
import pandas as pd
from environment import Cube
from model import ActorCritic

class Evaluator():
    def __init__(self, test_cubes_path="test_cubes.pkl", device="cpu"):
        self.test_cubes_path = test_cubes_path
        self.device = device
        self.best_score = 0
        self.wandb = False
        self.run_name = "model"
        self.keep_n_models = 5

    def evaluate(self, model, device):
        with open(f"{self.test_cubes_path}", "rb") as f:
            test_cubes = pickle.load(f)

        num_cubes = len(test_cubes)
        output = {l: 0 for l in range(1, num_cubes + 1)}
        table = []
        for level, cubes in test_cubes.items():
            # for cube in cubes:
            cubes = [Cube(cube.state) for cube in cubes]
            for i in range(level+1):
                states = np.array([cube.state for cube in cubes])
                policies, values = model(torch.Tensor(states).long().to(device))
                table.append([level, i, values.mean().item(), policies.max(dim=1)[0].mean().item()])
                moves = policies.argmax(dim=1).cpu().numpy()
                for cube, move in zip(cubes, moves):
                    cube.step(move)
                    if cube.is_solved():
                        output[level] += 1
                        cubes.remove(cube)

        score = sum([output[l] * l for l in output]) / (len(test_cubes) * len(test_cubes[1]))
        return output, score, table
    
    def evaluate_level(self, model, device, level):
        with open(f"{self.test_cubes_path}", "rb") as f:
            test_cubes = pickle.load(f)

        cubes = test_cubes[level]
        output = 0
        for cube in cubes:
            cube = Cube(cube.state)
            for i in range(level+1):
                with torch.no_grad():    
                    move = model(torch.Tensor(cube.state).long().to(device).unsqueeze(0))[0].argmax().item()
                cube.step(move)
                if cube.is_solved():
                    output += 1
                    break
        return output/len(cubes)
    
    def set_queue(self, queue):
        self.queue = queue

    def add_model(self, state_dict):
        self.queue.put(copy.deepcopy(state_dict))

    def init_wandb(self, run_id, project):
        self.wandb = True
        self.run_id = run_id
        self.project = project

    def run(self):
        if self.wandb:
            wandb.init(project=self.project, id=self.run_id, resume='allow')
            model_name = wandb.run.name
        else:
            model_name = "model"

        if os.path.exists("scores.json"):
            with open("scores.json", "r") as f:
                scores = json.load(f)

        else:
            scores = {}

        if not scores.get(model_name):
            scores[model_name] = {}

        model = ActorCritic()
        df = pd.DataFrame(columns=["episode", "run", "level", "iteration", "mean-value", "max-policy"])
        score_df = pd.DataFrame()
        previous_episode = None

        while True:
            if self.queue.qsize() == 0:
                continue

            queue_item = self.queue.get()
            if queue_item == "STOP":
                break

            state_dict, episode = queue_item
            if not previous_episode == episode:
                run = 1
                previous_episode = episode
            else:
                run += 1

            if not scores[model_name].get(episode):
                scores[model_name][episode] = 0

            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            output, score, table = self.evaluate(model, self.device)
            table = pd.DataFrame(table, columns=["level", "iteration", "mean-value", "max-policy"])
            table["episode"] = episode
            table["run"] = run
            output_df = pd.DataFrame(output.items(), columns=["level", "solved"])
            output_df["episode"] = episode
            output_df["run"] = run
            output_df["score"] = score
            score_df = pd.concat([score_df, output_df])
            score_df.to_csv(f"results/{model_name}_scores.csv", index=False)
    
            df = pd.concat([df, table])
            df.to_csv(f"results/{model_name}.csv", index=False)

            max_scores = sorted(scores[model_name].items(), key=lambda x: x[1], reverse=True)[:self.keep_n_models]
            models_saved = [x for x in os.listdir("models") if f"{model_name}_episode" in x]
            for m in models_saved:
                ep = int(m.split("_")[-2])
                if ep not in [int(x[0]) for x in max_scores] and ep != episode:
                    os.remove(f"models/{m}")

            torch.save(model.state_dict(), f"models/{model_name}_latest.pt")
            
            if score > scores[model_name][episode]:
                scores[model_name][episode] = score
                torch.save(model.state_dict(), f"models/{model_name}_episode_{episode}_best.pt")
            
            with open("scores.json", "w") as f:
                json.dump(scores, f)

            if self.wandb:
                wandb.log({"score": score, "output": output})
                # columns = ["level", "iteration", "mean-value", "max-policy"]
                # wandb.log({"table": wandb.Table(data=table, columns=columns)})