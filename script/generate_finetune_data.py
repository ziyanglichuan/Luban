import json
import pandas as pd

def finetune_process():
    input_file = '../finetune_data/alpaca_gpt4_data_zh.json'
    output_file = '../finetune_data/finetune_data.csv'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = [
        {
            'prompt': per['instruction'] + per['input'],
            'answer': per['output']
        }
        for per in data
        if 10 <= len(per['instruction'] + per['input']) <= 256 and 5 <= len(per['output']) <= 256
    ]

    df = pd.DataFrame(filtered_data)
    df.to_csv(output_file, index=False)
    print(df)

if __name__ == "__main__":
    finetune_process()
