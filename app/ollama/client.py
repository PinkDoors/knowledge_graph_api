import os
import json
import requests

BASE_URL = os.environ.get('OLLAMA_HOST', 'http://host.docker.internal:11434')

def generate(model_name, prompt, system=None, template=None, context=None, options=None, callback=None):
    try:
        url = f"{BASE_URL}/api/generate"

        if options is None:
            options = {}
        options["temperature"] = 0

        payload = {
            "model": model_name,
            "prompt": prompt,
            "system": system,
            "template": template,
            "context": context,
            "options": options
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            final_context = None
            full_response = ""

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)

                    if callback:
                        callback(chunk)
                    else:
                        if not chunk.get("done"):
                            response_piece = chunk.get("response", "")
                            full_response += response_piece
                            print(response_piece, end="", flush=True)

                    if chunk.get("done"):
                        final_context = chunk.get("context")

            return full_response, final_context

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None

