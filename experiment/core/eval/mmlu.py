import torch

from core.data.get_validation_dataset import get_mmlu_dataset_df

from utils import get_next_word_predictions, create_prompt_with_tulu_chat_format

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval_hf_model_generate_ICL_prompts(args, model, tokenizer, dev_df, test_df, batch_size=1):

    subject = args.subject

    args.use_chat_format = True
    k = args.n_val

    prompts = []

    chat_formatting_function = create_prompt_with_tulu_chat_format

    for i in range(0, test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is:"
            else:
                prompt += " The answer is:"

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids

        # make sure every prompt is less than 2048 tokens
        while len(tokenized_prompt) > 512:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"

            tokenized_prompt = tokenizer(
                prompt, truncation=False, add_special_tokens=False).input_ids
        prompts.append(prompt)

    return prompts


@torch.no_grad()
def compute_accuracy(args, model, tokenizer, answer_choice_ids, batch_size=1):

    dev_df = get_mmlu_dataset_df(data_dir='./data',
                                 validation=True,
                                 k = args.n_val,
                                 subject = args.subject)

    eval_df = get_mmlu_dataset_df(data_dir='./data',
                                 validation=False,
                                 k = args.n_eval,
                                 subject = args.subject)

    prompts = eval_hf_model_generate_ICL_prompts(args, model, tokenizer, dev_df, eval_df, batch_size=1)

    # for prompt in prompts:
    #     print('')
    #     print('*** ICL Prompt Starts ***')
    #     print(prompt)
    #     print('*** ICL Prompt Ends ***')

    # # get the answer for all examples
    # # adding a prefix space here, as that's expected from the prompt
    # # TODO: should raise a warning if this returns more than one token
    # answer_choice_ids = [tokenizer.encode(
    #     " " + answer_choice, add_special_tokens=False)[-1] for answer_choice in choices]

    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False, batch_size=batch_size
    )

    # get the metrics
    cors = []
    groud_truths = eval_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)

    acc = torch.tensor(cors, dtype=torch.float32).mean().item()
    cors = torch.tensor(cors)

    all_probs = torch.tensor(all_probs)
    return cors, acc, all_probs