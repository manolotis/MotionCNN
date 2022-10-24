import sys

sys.path.append("/home/manolotis/sandbox/robustness_benchmark/")
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm

from train import WaymoLoader, pytorch_neg_multi_log_likelihood_batch
from motionCNN.code.utils.predict_utils import parse_arguments, get_config

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

args = parse_arguments()
config = get_config(args)

model_name = config["model"]["name"]
if "name_addition" in config["model"]:
    model_name = config["model"]["name"] + "_" + config["model"]["name_addition"]

savefolder = os.path.join(config["test"]["output_config"]["out_path"], model_name)

if not os.path.exists(savefolder):
    os.makedirs(savefolder, exist_ok=True)

model_path = os.path.join(config["model"]["path"], config["model"]["name"], "model_best.pt")
model = torch.jit.load(model_path).cuda().eval()
loader = DataLoader(
    WaymoLoader(config["test"]["data_config"]["dataset_config"]["data_path"], return_vector=True),
    batch_size=1,
    num_workers=1,
    shuffle=False,
)

with torch.no_grad():
    for x, y, is_available, vector_data, extra_data in tqdm(loader):
        # for x, y, is_available, vector_data in loader:
        # x: images, y: ground truth future trajectories

        x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

        confidences_logits, logits = model(x)

        argmax = confidences_logits.argmax()

        loss = pytorch_neg_multi_log_likelihood_batch(
            y, logits, confidences_logits, is_available
        )
        confidences = torch.softmax(confidences_logits, dim=1)
        V = vector_data[0]

        logits = logits.squeeze(0).cpu().numpy()
        y = y.squeeze(0).cpu().numpy()
        is_available = is_available.squeeze(0).long().cpu().numpy()
        confidences = confidences.squeeze(0).cpu().numpy()

        scenario_id = extra_data["scenario_id"][0]
        agent_id = extra_data["agent_id"].item()
        agent_type = extra_data["agent_type"].item()

        filename = f"scid_{scenario_id}__aid_{agent_id}__atype_{agent_type}.npz"
        savedata = {
            "scenario_id": scenario_id,
            "agent_id": agent_id,
            "agent_type": agent_type,
            "coordinates": logits,
            "probabilities": confidences,
            # "target/history/xy": data_original["target/history/xy"][agent_index],
            "target/future/xy": y,
            # "target/history/valid": data_original["target/history/valid"][agent_index],
            "target/future/valid": is_available.reshape((is_available.shape[0], 1))
        }
        np.savez_compressed(os.path.join(savefolder, filename), **savedata)
