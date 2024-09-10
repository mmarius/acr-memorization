import json
import logging
import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import prompt_optimization as prompt_opt
from prompt_optimization.utils import get_id_func, now, load_target_str

OmegaConf.register_new_resolver("generate_id", get_id_func())


@hydra.main(version_base=None, config_path="config", config_name="promptmin")
def main(cfg):
    # Set randomness
    if cfg.seed:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    for arg, value in OmegaConf.to_container(cfg, resolve=True).items():
        logging.info(f"{arg}: {value}")

    # Device, model, and tokenizer setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        model_args = dict(
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32, # TODO(mm): using bfloat16 here will lead to issues when generating. Results won't be consistent with argmax decoding
            # device_map="auto",
        )
    else:
        model_args = dict(
            trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float32
        )
    # model_args = dict(trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_args)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # if torch.cuda.device_count() <= 1:
    model = model.to(device)

    if cfg.random_weights:
        logging.info("Randomizing weights")
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

    # Data setup
    input_str = cfg.input_str
    target_str = cfg.target_str
    chat_template = cfg.chat_template
    system_prompt = cfg.system_prompt

    if cfg.dataset is not None and cfg.data_idx is not None:
        target_str = load_target_str(cfg.dataset, cfg.data_idx, tokenizer)
        cfg.target_str = target_str

    # Add space at the beginning of the target string
    target_str = " " + target_str  # TODO(mm): this should potentially be model specific
    logging.info(
        f"Target string: {target_str}"
    )

    # Optimization setup
    optimization_args = {
        "discrete_optimizer": cfg.discrete_optimizer,
        "num_steps": cfg.num_steps,
        "lr": cfg.lr,
        "optimizer": cfg.optimizer,
        "batch_size": cfg.batch_size,
        "mini_batch_size": cfg.mini_batch_size,
        "topk": cfg.topk,
    }

    solution = prompt_opt.minimize_prompt(
        model,
        tokenizer,
        input_str,
        target_str,
        system_prompt,
        chat_template,
        device,
        optimization_args,
        n_tokens_in_prompt=cfg.n_tokens_in_prompt,
        max_tokens=cfg.max_tokens,
        stop_early=cfg.stop_early,
    )
    input_slice, target_slice, loss_slice, input_ids = (
        solution["input_slice"],
        solution["target_slice"],
        solution["loss_slice"],
        solution["input_ids"],
    )

    # Test the prompt and log the new generation with the target string
    if solution["success"] is True:
        logging.info("---------------------- Optimization DONE ----------------------")

        optimized_ids = solution["input_ids"]  # that's the full sequence

        prompt = optimized_ids[input_slice]
        logging.info(f"Optimized prompt (ids): {prompt.tolist()}")

        prompt_decoded = tokenizer.decode(
            prompt[input_slice], skip_special_tokens=False
        )
        prompt_tokens = tokenizer.convert_ids_to_tokens(prompt)
        logging.info(f"Optimized prompt (tokens): {prompt_tokens}")
        logging.info(f"Optimized prompt: {prompt_decoded}")

        # TODO(mm): in some case generate and argmax decoding do not return the same output
        output = model.generate(
            input_ids=prompt.unsqueeze(0),
            attention_mask=torch.ones(len(prompt)).unsqueeze(0).to(device),
            max_new_tokens=len(
                optimized_ids[target_slice]
            ),  # generate as many tokens as the target has
            do_sample=False,
            num_beams=1,
        )

        logging.info(
            f"goal: {tokenizer.decode(optimized_ids[target_slice], skip_special_tokens=False)}"
        )
        logging.info(
            f"output (generated): {tokenizer.decode(output[0, target_slice], skip_special_tokens=False)}"
        )

        gen_matches_target = torch.all(
            output[0, target_slice] == optimized_ids[target_slice]
        ).item()
        logging.info(f"generation matches target: {gen_matches_target}")

        # Calculate loss for the target_ids
        with torch.no_grad():
            ids_for_loss_computation = input_ids[target_slice].unsqueeze(0).to(device)
            outputs = model(ids_for_loss_computation, labels=ids_for_loss_computation)
        loss_of_target_str = outputs.loss.item()

        with torch.no_grad():
            ids_for_loss_computation = input_ids[input_slice].unsqueeze(0).to(device)
            outputs = model(ids_for_loss_computation, labels=ids_for_loss_computation)
        loss_of_prompt = outputs.loss.item()

        solution["input_ids"] = input_ids.tolist()

        # Compile data for saving to a JSON file
        results = {
            "target_length": target_slice.stop - target_slice.start,
            "target_str": target_str,
            "loss_of_target_str": loss_of_target_str,
            "loss_of_prompt": loss_of_prompt,
            "success": True,
            "optimal_prompt": prompt_decoded,
            "optimal_prompt_ids": prompt.tolist(),
            "optimal_prompt_tokens": prompt_tokens,
            "gen_matches_target": gen_matches_target,
        }
        for k, v in solution.items():
            if isinstance(v, slice):
                results[k] = (v.start, v.stop)
            else:
                results[k] = v
    else:
        results = {
            "success": False,
            "num_free_tokens": solution["num_free_tokens"],
            "target_str": target_str,
            "target_length": target_slice.stop - target_slice.start,
        }

    for k, v in OmegaConf.to_container(cfg, resolve=True).items():
        results[f"cfg_{k}"] = v

    # log data to the console
    for key, value in results.items():
        logging.info(f"{key}: {value}")
    results["time"] = now()

    # Save the data to a JSON file
    filename = os.path.join(HydraConfig.get().run.dir, "results.json")
    with open(filename, "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    main()
