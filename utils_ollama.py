import urllib.request
import json


# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion


def query_model(
    prompt="",
    model="llama3.2",
    url="http://localhost:11434/api/chat",
    role="user",
    messages=None,
    seed=123,
    temperature=0.0,
    tools=None,
    num_predict=256,
    num_ctx=4096,
):

    # Create the data payload as a dictionary

    options = {  # Some settings below are required for deterministic responses
        "seed": seed,  # alone setted to a fixed number (different from 0) it allows to generate the same constant output text (for the same prompt). Set to 0 for costant outputs.
        "temperature": temperature,  # alone setted to 0 does not ensure a constant output text
        "num_ctx": num_ctx,  # max input token
        "top_p": 0.9,
        "stop": ["<|eot_id|>", "<|eom_id|>"],
        "num_predict": num_predict,  # max output tokens
    }
    stream = False  # "stream"

    if not (messages):  # for basic message with a single role eg. user or assistant
        data = {
            "model": model,
            "stream": stream,
            "options": options,
            "messages": [{"role": role, "content": prompt}],
        }

    else:  # advance messages settings
        data = {
            "model": model,
            "stream": stream,
            "options": options,
            "messages": messages,
        }

    # https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-tools
    if tools is not None:
        data["tools"] = tools
        data["stream"] = False

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)

            if not response_json["message"].get("tool_calls"):
                response_data += response_json["message"]["content"]
                # print(
                #    "The model doesn't use the function. Its response is:",
                #    str(response_data),
                # )

            else:
                response_data = response_json["message"]  # ["tool_calls"]
                print(
                    "The model use the function. Its response is:", str(response_data)
                )

    return response_data


def query_model_embed(
    input="",
    model="llama3.2",
    url="http://localhost:11434/api/embed",
):

    data = {"model": model, "input": input}

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)

            # print("response_json:",response_json)

            response_data = response_json["embeddings"][0]

    return response_data


def built_messages(
    user_prompt_custom="", system_prompt_content="", user_prompt_static=""
):

    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_prompt_static + user_prompt_custom},
    ]
    return messages
