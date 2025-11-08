import json, os
import editdistance
import argparse

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--data_file", type=str, default="")
    args.add_argument("--output_file", type=str, default="")
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    data = []
    file = []
    with open(args.data_file) as f:
        file += [json.loads(line) for line in f]

    for item in file:
        if item['step_margin'] < 0.5 or editdistance.eval(item['rejected_step'], item['chosen_step']) < 20:
            continue
        data.append({
            "prompt": item['prefix'],
            "neg": item['rejected_step'],
            "pos": item['chosen_step'],
            "neg_count": item['neg_count'],
            "pos_count": item['pos_count'],
        })

    print(len(data))

    with open(args.output_file, 'w') as f:
        json.dump(data, f, indent=4)