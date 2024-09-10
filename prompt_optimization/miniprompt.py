"""
miniprompt.py
an implementation of miniprompt

developed in collaboration by: Avi Schwarzschild and Zhili Feng and Pratyush Maini in 2024
"""

import logging

import prompt_optimization as prompt_opt


def minimize_prompt(
    model,
    tokenizer,
    input_str,
    target_str,
    system_prompt,
    chat_template,
    device,
    optimization_args,
    n_tokens_in_prompt = 10, # number of placeholder tokens to start with
    max_tokens=30, # maximum number of tokens to try
    stop_early=False, # whether to stop as soon as a valid prompt is found
):
    running_max = max_tokens
    running_min = 0
    success = False
    best_input = None
    best_prompt = None
    done = False
    best_slices = (None, None, None, None)

    while not done:
        logging.info("\n------------------------------------\n")
        logging.info(f"{n_tokens_in_prompt} tokens in the prompt")
        input_ids, free_token_slice, input_slice, target_slice, loss_slice = (
            prompt_opt.prep_text(
                input_str,
                target_str,
                tokenizer,
                system_prompt,
                chat_template,
                n_tokens_in_prompt,
                device,
            )
        )
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        logging.info(f"input_tokens: {input_tokens}")
        logging.info(f"input_tokens (free_token_slice): {input_tokens[free_token_slice]}")
        logging.info(f"input_tokens (input_slice): {input_tokens[input_slice]}")
        logging.info(f"input_tokens (target_slice): {input_tokens[target_slice]}")
        logging.info(f"input_tokens (loss_slice): {input_tokens[loss_slice]}")

        # encode the input sequence to create an attention mask
        # input_sequence = tokenizer.decode(input_ids, skip_special_tokens=False)
        # encoded_input = tokenizer(input_sequence, return_tensors="pt", add_special_tokens=False)
        # attention_mask = encoded_input["attention_mask"].to(device)

        if running_max == -1:
            running_max = (target_slice.stop - target_slice.start) * 5
        if optimization_args["discrete_optimizer"] == "gcg":
            solution = prompt_opt.optimize_gcg(
                model,
                tokenizer,
                input_ids,
                input_slice,
                free_token_slice,
                target_slice,
                loss_slice,
                optimization_args["num_steps"],
                batch_size=optimization_args["batch_size"],
                topk=optimization_args["topk"],
                mini_batch_size=optimization_args["mini_batch_size"],
            )
        elif optimization_args["discrete_optimizer"] == "random_search":
            solution = prompt_opt.optimize_random_search(
                model,
                input_ids,
                input_slice,
                free_token_slice,
                target_slice,
                loss_slice,
                optimization_args["num_steps"],
                batch_size=optimization_args["batch_size"],
                mini_batch_size=optimization_args["mini_batch_size"],
            )
        else:
            raise ValueError(
                "discrete_optimizer must be one of ['gcg', 'random_search']"
            )

        # check whether the completion matches the target 
        target_acquired = prompt_opt.check_output_with_hard_tokens(
            model, solution["input_ids"].unsqueeze(0), target_slice, loss_slice
        )

        if target_acquired:
            logging.info(
                f"Target acquired with {n_tokens_in_prompt} tokens in the prompt"
            )
            running_max = n_tokens_in_prompt
            success = True
            best_input = solution["input_ids"]
            best_prompt = solution["input_ids"][input_slice]
            best_prompt_tokens = tokenizer.convert_ids_to_tokens(best_prompt)
            best_prompt_text = tokenizer.decode(best_prompt, skip_special_tokens=False)
            logging.info(f"Prompt found (ids): {best_prompt.tolist()}")
            logging.info(f"Prompt found (tokens): {best_prompt_tokens}")
            logging.info(f"Prompt found: {best_prompt_text}")
            new_num_tokens = n_tokens_in_prompt - 1
            best_slices = (free_token_slice, input_slice, target_slice, loss_slice)

            if stop_early:
                done = True # stop as soon as a valid prompt is found

        else:
            logging.info(
                f"Target NOT acquired with {n_tokens_in_prompt} tokens in the prompt"
            )
            new_num_tokens = n_tokens_in_prompt + 5
            running_min = n_tokens_in_prompt
            optimization_args["num_steps"] = int(optimization_args["num_steps"] * 1.2)

        if (new_num_tokens >= running_max) or (new_num_tokens <= running_min):
            done = True
        else:
            n_tokens_in_prompt = new_num_tokens

    output = {
        "free_token_slice": best_slices[0]
        if best_slices[0] is not None
        else free_token_slice,
        "input_slice": best_slices[1] if best_slices[1] is not None else input_slice,
        "target_slice": best_slices[2] if best_slices[2] is not None else target_slice,
        "loss_slice": best_slices[3] if best_slices[3] is not None else loss_slice,
        "success": success,
        "num_free_tokens": running_max,
        "input_ids": best_input,
    }
    return output
