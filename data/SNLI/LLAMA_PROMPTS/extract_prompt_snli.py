import json
def read_prompt_file(json_file_path):
        with open(json_file_path,'r') as file:
                prompts = json.load(file)

        for key in prompts:
                prompt = prompts[key]
                print(prompt)
                return

if __name__ == "__main__":
        read_prompt_file(
                '/Users/sbhar/Riju/PhDCode/ACL/data/SNLI/llama_predictions/negC_negH_pred.json'
        )