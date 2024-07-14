from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Phi-3-mini-128k-instruct-8bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# messages = [
#     {"role": "system", "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."},
#     {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
#     {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
#     {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
# ]

# Need to test using the non mlx version and clock it
# response = generate(model, tokenizer, prompt=messages[0], verbose=True)

# List of prompts
prompts = [
    'You are an expert on COVID facts. Can you tell me whether the following statement is true, or false: Drinking lemon juice cures COVID in 24 hours. Please response with "True" or "False".', 
    'You are an expert on COVID facts. Can you tell me whether the following statement is true, or false: Covid was created in a lab by Obama. Please response with "True" or "False".', 
    'You are an expert on COVID facts. Can you tell me whether the following statement is true, or false: Trump cured everyone from COVID. Please response with "True" or "False".' 
]

# List to store responses
responses = []

# Loop over each prompt and call the generate function
for prompt in prompts:
    response = generate(model, tokenizer, prompt=prompt, verbose=False, 
                        max_tokens=100)
    responses.append(response)

responses[0]
# Now responses contains the output for each prompt
for i, response in enumerate(responses):
    print(f"Response to '{prompts[i]}': {response}")


def parse_output(output):
    # Define the tags you are looking for
    tags = [" True", " False"]

    # Find the position of the first occurrence of any tag and extract the statement
    for tag in tags:
        tag_pos = output.find(tag)
        if tag_pos != -1:
            # Extract the statement right after the tag
            end_of_statement = output.find('.', tag_pos)
            if end_of_statement != -1:
                return output[tag_pos + len(tag): end_of_statement].strip()
    
    return "No statement found"  # Return this if no tags are found

# Initialize responses as a dictionary
responses = {}

# Loop over each prompt and process responses
for prompt in prompts:
    raw_response = generate(model, tokenizer, prompt=prompt, verbose=True)
    parsed_response = parse_output(raw_response)
    responses[prompt] = parsed_response  # This should now work without error

# Outputting the collected responses
