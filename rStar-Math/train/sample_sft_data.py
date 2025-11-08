import json
import random
import argparse
random.seed(0)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--data_file", type=str, default="")
    args.add_argument("--n", type=int, default=2)
    args.add_argument("--output_file", type=str, default="")
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    data = []

    with open(args.data_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass

    PR_dict = {}
    MAX_SIZE = args.n

    for item in data:
        prompt = item['question']
        response = item['sft_full']
        if prompt not in PR_dict:
            PR_dict[prompt] = []
        avg = sum(item['q_values']) / len(item['q_values']) if item['q_values'] else 0
        mmin = min(item['q_values']) if item['q_values'] else 1
        if len(item['q_values']) <= 1:
            mmax = -1
        else:
            mmax = max(item['q_values'][:-1])
        PR_dict[prompt].append({
            'response': response,
            'Q': item.get('final_Q', 1),
            "avg": avg,
            "min": mmin,
            "max": mmax,
            "pred": item.get('ORM_pred', 0)
            }
        )

    PR = []
    Q_count = 0

    for prompt, responses in PR_dict.items():
        ques = prompt.strip().replace("<|user|>:\n", "").strip()
        corrects = [
            {
                'res': item['response'],
                'avg': item['avg'],
                'min': item['min'],
                'max': item['max'],
                'pred': item['pred']
            } for item in responses if item['Q'] == 1
        ]

        if corrects:
            corrects = sorted(corrects, key=lambda x: (x['avg'], x['min']), reverse=True)
            corrects = [item['res'] for item in corrects]

        if len(corrects) > MAX_SIZE:
            corrects = corrects[:MAX_SIZE]

        for item in corrects:
            if "error" in item.lower():
                continue
            PR.append({
                'query': prompt.replace('<|user|>:\n', ''),
                'response': item.replace("<|assistant|>: Let's think step by step and solve the problem with code.", '').strip()
            })

    random.shuffle(PR)

    print(len(PR))
    with open(args.output_file, 'w') as f:
        json.dump(PR, f, indent=4)