import argparse
import logging
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_texts():
    texts = [
        "this is a first prompt", 
        "this is a second prompt"
    ]
    return texts


def main(args):
    # Set randomness
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Generate a unique ID for the run and create the experiments directory
    run_id = "generate_dummy"
    os.makedirs("outputs/", exist_ok=True)

    # Setup logging configuration
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f"outputs/{run_id}.log"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"run id: {run_id}")

    # log arguments
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    # Device, model, and tokenizer setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model_args = dict(
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,  # TODO(mm): using bfloat16 here will lead to issues when generating. Results won't be consistent with argmax decoding
            # device_map="auto", # this will put different layers on differt GPUs (only use for large models)
        )
    else:
        model_args = dict(trust_remote_code=False, low_cpu_mem_usage=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # set pad token to eos token for models that don't have a padding token
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)  # don't use .to(device) if using device_map="auto"

    # Load data
    texts = _get_texts()

    distributions = None 
    # Generate text from the model
    encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(**encoding, max_new_tokens=20, return_dict_in_generate=True, output_scores=True)
        batch_logits = outputs.scores
        for logits in batch_logits:
            distributions = (
                np.concatenate([distributions, logits[None, :].cpu().detach().numpy()], axis=0)
                if distributions is not None
                else logits[None, :].cpu().detach().numpy()
            )

    distributions = np.swapaxes(distributions, 0, 1)  # swap token and bsz axis
    print(distributions.shape)

    # Save results to disk
    output_dir = f"outputs/{args.model_name.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)  # create output directory
    np.save(f"{output_dir}/{run_id}_distributions.npy", distributions)


if __name__ == "__main__":
    # Setup argument parser to get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="experiments")
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--data-source", type=str, default=None)
    parser.add_argument("--target-str", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    # Run the main function
    main(args)
