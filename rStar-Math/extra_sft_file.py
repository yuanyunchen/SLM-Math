import os
from pebble import ProcessPool
from functools import partial
from eval_output import *
from rstar_deepthink.agents.utils import *


def extract_boxed_answer(text, debug=False):
    if text is None:
        return None
    start = text.rfind(r"boxed{")
    if start == -1:
        start = text.rfind(r"answer{")
    if start == -1:
        return text
    end = None
    stack = []
    answer = text[start:]
    for i, c in enumerate(answer):
        if c == "{":
            stack.append(i)
        elif c == "}":
            start = stack.pop()  # \boxed start{
            if len(stack) == 0:
                end = i  # \boxed end}
                break
    if end is None and debug:
        print("brack not closing", answer)
        return None
    return answer[start + 1 : end]


def search_all_traces(node: InferNode) -> List[List[InferNode]]:
    tmp_list = []
    ret_list = []
    def dfs(node: InferNode):
        if not node.children:
            tmp_list.append(node)
            ret_list.append(tmp_list.copy())
            tmp_list.pop()
            
        tmp_list.append(node)
        for child in node.children:
            dfs(child)
        tmp_list.pop()
    
    dfs(node)

    return ret_list


def build_solution(valid_traces: List[List[InferNode]], wrong_traces: List[List[InferNode]], ground_truth):
    correct_steps = []
    question= None
    
    for trace in valid_traces:
        question = "<|user|>:\n" + trace[0].extra_info[10:] + "\n<|assistant|>: Let's think step by step and solve the problem with code."
        extra_ans = trace[-1].final_answer
        full = question
        q_values, values, visit_counts, pucts = [], [], [], []
        sft_full = ""
        prm_full = []
        for idx in range(1, len(trace)):
            sft_full += trace[idx].text
            full += trace[idx].text
            q_values.append(trace[idx].q_value)
            values.append(trace[idx].value)
            visit_counts.append(trace[idx].visit_count)
            pucts.append(trace[idx].puct())
            prm_full.append(
                    {
                        "step": trace[idx].text,
                        "Q": trace[idx].q_value,
                    }
                )
        
        trace_dict = {
            "full": full,
            "question": trace[0].extra_info[10:],
            "final_Q": 1,
            "q_values": q_values,
            "values": values,
            "visit_counts": visit_counts,
            "pucts": pucts,
            "gt": ground_truth,
            "extra_ans": extra_ans,
            "sft_full": sft_full,
            "prm_full": json.dumps([{"content": prm_full}]),
        }
        correct_steps.append(trace_dict)

    wrong_steps = []
    for trace in wrong_traces:
        question = "<|user|>:\n" + trace[0].extra_info[10:] + "\n<|assistant|>: Let's think step by step and solve the problem with code."
        extra_ans = trace[-1].final_answer
        full = question
        q_values, values, visit_counts, pucts = [], [], [], []
        sft_full = ""
        prm_full = []
        for idx in range(1, len(trace)):
            sft_full += trace[idx].text
            full += trace[idx].text
            q_values.append(trace[idx].q_value)
            values.append(trace[idx].value)
            visit_counts.append(trace[idx].visit_count)
            pucts.append(trace[idx].puct())
            prm_full.append(
                    {
                        "step": trace[idx].text,
                        "Q": trace[idx].q_value,
                    }
                )
        
        trace_dict = {
            "full": full,
            "question": trace[0].extra_info[10:],
            "final_Q": -1,
            "q_values": q_values,
            "values": values,
            "visit_counts": visit_counts,
            "pucts": pucts,
            "gt": ground_truth,
            "extra_ans": extra_ans,
            "sft_full": sft_full,
            "prm_full": json.dumps([{"content": prm_full}]),
        }
        wrong_steps.append(trace_dict)

    return wrong_steps + correct_steps


def extra_solution_dict( 
    full_tree_dict: Dict[str, Any], 
    prune: bool = True,
    b1: int = 64,
    b2: int = 16,
    c_puct: float = 2,
) -> Optional[Dict[str, Any]]:
    ground_truth = full_tree_dict.get("answer", None)
    tree_dict = full_tree_dict["rstar"]
    # rebuild tree
    root, tree_depth = rebuild_tree(tree_dict, max_num_children=b1*b2, c_puct=c_puct)

    # pruning tree
    if prune:
        prune_node(root)
        if root.prune:
            # no valid leaf node for the entire tree
            return []
    
    traces = search_all_traces(root)
    valid_traces = []
    invalid_traces = []
    for trace in traces:
        if is_valid_final_answer_node(trace[-1]) and math_equiv(trace[-1].final_answer, ground_truth):
            valid_traces.append(trace)
        elif is_valid_final_answer_node(trace[-1]):
            invalid_traces.append(trace)
        elif not trace[-1].children and trace[-1].final_answer and trace[-1].final_answer not in [NO_VALID_CHILD, TOO_MANY_STEPS]:
            # trace with code error
            invalid_traces.append(trace)
    res =  build_solution(valid_traces, invalid_traces, ground_truth)
    return res


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="")
    args.add_argument("--output_file", type=str, default="")
    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cnt, total = 0, 0
    solutions = []
    dirs = args.data_dir
    full_tree_dicts = []
    for dir in os.listdir(dirs):
        if not dir.endswith(".jsonl"): continue
        if "extra" in dir: continue
        with open(dirs + "/" + dir, "r") as f:
            for line in f:
                try:
                    full_tree_dict = json.loads(line)
                    full_tree_dicts.append(full_tree_dict)
                except:
                    pass
    
    def get_one_solution(full_tree_dict):
        solution = extra_solution_dict(
                    full_tree_dict
                )
        return solution
        
    
    def save_batch_to_jsonl(solutions, file_path):
        with open(file_path, "a") as f:
            for solution in solutions:
                json.dump(solution, f)
                f.write("\n")
    batch_size = 5000
    output_file = args.output_file
    
    with ProcessPool(max_workers=os.cpu_count() - 8) as pool:
        executor = partial(get_one_solution)
        
        for i in range(0, len(full_tree_dicts), batch_size):
            batch = full_tree_dicts[i:i + batch_size]
            future = pool.map(executor, batch, timeout=50)
            iterator = future.result()

            progress_bar = tqdm(total=len(batch), desc=f"Processing batch {i // batch_size + 1}")

            batch_solutions = []
            while True:
                try:
                    result = next(iterator)
                    batch_solutions.extend(result)
                except StopIteration:
                    break
                except Exception as error:
                    print(error)

                if progress_bar is not None:
                    progress_bar.update(1)

            if progress_bar is not None:
                progress_bar.close()

            save_batch_to_jsonl(batch_solutions, output_file)

    print("All batches processed.")