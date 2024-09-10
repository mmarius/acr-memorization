import argparse
import logging
import os

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader


def _get_dataset_from_csv(data_source):
    # Load data from .csv file
    if data_source.endswith(".csv"):
        df = pd.read_csv(data_source, sep="\t")
        return Dataset.from_pandas(df)
    else:
        raise ValueError("Data source must be a .csv file.")


def _filter_by_target(dataset, target_str):
    # Filter data by target string
    return dataset.filter(lambda e: e["target_str"] == target_str)


def _create_dataloader(dataset, tokenizer, ignore_prompts=False, batch_size=5):
    # Given a list of prompts and corresponding target strings, create samples
    # TODO(mm): create a dataloader that returns batches of samples

    # First we need to convert the prompt ids to list of ints
    def _update_prompt_ids(examples):
        examples["optimal_prompt_ids"] = eval(examples["optimal_prompt_ids"])
        return examples

    dataset = dataset.map(_update_prompt_ids, batched=False)

    def _construct_full_sequence(example):
        # TODO(mm): Implement a version that can be run in batched mode
        prompt_ids = example["optimal_prompt_ids"]
        target_ids = tokenizer(example["target_str"], add_special_tokens=False)["input_ids"]
        if not ignore_prompts:        
            example["full_sequence"] = prompt_ids + target_ids
        else:
            example["full_sequence"] = [tokenizer.bos_token_id] + target_ids
        return example
    dataset = dataset.map(_construct_full_sequence, batched=False) # this needs to be run with batched=False

    def _encode(examples):
        results = tokenizer.pad(
            {"input_ids": examples["full_sequence"]}, # this is a list of tokens
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        results["num_free_tokens"] = examples["num_free_tokens"]
        
        target_slices = []
        for idx, _ in enumerate(examples["optimal_prompt_ids"]):
            if not ignore_prompts:
                target_slices.append(eval(examples["target_slice"][idx]))
            else:
                target_slices.append((1, examples["target_length"][idx]))
        results["target_slice"] = target_slices

        return results

    dataset = dataset.map(_encode, batched=True, batch_size=10)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "target_slice"])

    if batch_size > len(dataset):
        batch_size = len(dataset)

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


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
    run_id = "generate_target"
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

    # Load data from .csv
    if args.data_source:
        dataset = _get_dataset_from_csv(args.data_source)
        if args.target_str:  # filter by target string
            dataset = _filter_by_target(dataset, args.target_str)
    else:
        raise ValueError("Data source must be provided.")

    samples = _create_dataloader(dataset, tokenizer, ignore_prompts=args.ignore_prompt)  # construct data loader

    distributions, target_slices = None, None 
    for batch in samples:
        # Run forward pass to get hidden states and logits
        outputs = model.forward(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        # outputs["hidden_states"] is a tuple with len=num_layers
        # each element is of shape: (bsz, seq_len, hidden_size)
        # print(outputs["hidden_states"][-1].shape)
        logits = outputs["logits"]  # (bsz, seq_len, vocab_size)

        # # DEBUGG ARGMAX DECODING
        # # --> results are correct! (23.08.2024)
        # batch_values, batch_indices = torch.topk(logits, k=10, dim=2)
        
        # # get values for target tokens only
        # for idx, values in enumerate(batch_values):
        #     indices = batch_indices[idx, :]
        #     target_indices = indices[batch["target_slice"][idx][0] - 1:batch["target_slice"][idx][1] - 1, :]
            
        #     for t in target_indices:
        #         output_tokens = tokenizer.convert_ids_to_tokens(t)
        #         print(output_tokens)
        #     print()
        # ###########################

        distributions = (
            np.concatenate([distributions, logits.cpu().detach().numpy()], axis=0)
            if distributions is not None
            else logits.cpu().detach().numpy()
        )

        target_slices = (
            np.concatenate([target_slices, batch["target_slice"].numpy()], axis=0)
            if target_slices is not None
            else batch["target_slice"].numpy()
        )

    # Save results to disk
    output_dir = f"outputs/{args.model_name.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)  # create output directory
    np.save(f"{output_dir}/{run_id}_distributions.npy", distributions)
    np.save(f"{output_dir}/{run_id}_target_slices.npy", target_slices)


if __name__ == "__main__":
    # Setup argument parser to get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="experiments")
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--data-source", type=str, default=None)
    parser.add_argument("--ignore-prompt", action="store_true")
    parser.add_argument("--target-str", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    # Run the main function
    main(args)
