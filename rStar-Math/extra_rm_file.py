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

def get_prefix(node: InferNode):
    prefix = ""
    while node.parent:
        prefix = node.text + prefix
        node = node.parent
    prefix = "<|user|>:\n" + node.extra_info[10:] + "\n<|assistant|>: Let's think step by step and solve the problem with code." + prefix
    return prefix

def search_all_traces(node: InferNode, mode="all"):
    ret_list = [] 

    def dfs(node: InferNode):
        for child in node.children:
            dfs(child)
        if is_valid_final_answer_node(node) or node.final_answer  == TOO_MANY_CODE_ERRORS:
            if node.q_value == 1:
                node.final_correct = 1
            elif node.q_value == -1 and node.final_answer != TOO_MANY_CODE_ERRORS:
                node.final_wrong = 1

        for child in node.children:
            node.final_correct += child.final_correct
            node.final_wrong += child.final_wrong
    
    dfs(node)
    
    question = "<|user|>:\n" + node.extra_info[10:] + "\n<|assistant|>: Let's think step by step and solve the problem with code."

    search_node = []
    if node:
        search_node = [node]
    
    while search_node:
        nodes = search_node[0].children
        
        chosen_candidates = [child for child in nodes if child.final_correct > 0]
        chosen_candidates = sorted(chosen_candidates, key=lambda x: x.q_value, reverse=True)
        
        rejected_candidates = [child for child in nodes if child.final_wrong > 0]
        rejected_candidates = sorted(rejected_candidates, key=lambda x: x.q_value)
        
        chosen_nodes = chosen_candidates[:2]
        rejected_nodes = rejected_candidates[:2]

        if not chosen_nodes or not rejected_nodes: 
            search_node = search_node[1:]
            continue

        if len(chosen_nodes) == 1: 
            rejected_nodes = rejected_candidates[:4]
        if len(rejected_nodes) == 1:
            chosen_nodes = chosen_candidates[:4]

        for chosen_node in chosen_nodes:
            for rejected_node in rejected_nodes:
                chosen_step = chosen_node.text
                rejected_step = rejected_node.text
                step_margin = chosen_node.q_value - rejected_node.q_value
                chosen_steps, chosen_steps_avg_score  = build_partial_solution(chosen_node, best=True)
                rejected_steps, rejected_steps_avg_score = build_partial_solution(rejected_node, best=False)
                if step_margin > 0 and chosen_steps_avg_score > rejected_steps_avg_score:
                    ret_list.append({
                        "prefix": get_prefix(search_node[0]),
                        "chosen_step": chosen_step,
                        "chosen_steps": chosen_steps,
                        "rejected_step": rejected_step,
                        "rejected_steps": rejected_steps,
                        "step_margin": step_margin,
                        "steps_margin": chosen_steps_avg_score - rejected_steps_avg_score,
                        "question": question,
                        "neg_count": len(rejected_nodes),
                        "pos_count": len(chosen_nodes),
                    })
        if mode == "vis_count":
            # choose max visit_count as the next search node
            nodes = sorted(nodes, key=lambda x: x.visit_count, reverse=True)
            for node in nodes:
                if node and node.final_correct > 0 and node.final_wrong > 0:
                    search_node.append(node)
                    break
        elif mode == "all":
            for node in nodes:
                if node and node.final_correct > 0 and node.final_wrong > 0:
                    search_node.append(node)
            
        search_node = search_node[1:]
    return ret_list

def build_partial_solution(node: InferNode, best: bool):
    ret_steps = []
    search_node = node
    while search_node:
        ret_steps.append(search_node)
        steps = []
        for child in search_node.children:
            if best and child.final_correct > 0:
                steps.append(child)
            if not best and child.final_wrong > 0:
                steps.append(child)
        if not steps: break
        steps = sorted(steps, key=lambda x: x.q_value, reverse=best)
        search_node = steps[0]
    chosen_steps = "".join([step.text for step in ret_steps])
    chosen_steps_avg_score = sum([step.q_value for step in ret_steps]) / len(ret_steps) if ret_steps else 0
    return chosen_steps, chosen_steps_avg_score


def extra_solution_dict( 
    full_tree_dict: Dict[str, Any], 
    prune: bool = True,
    b1: int = 64,
    b2: int = 16,
    c_puct: float = 2,
    mode: str = "all",
) -> Optional[Dict[str, Any]]:
    """
    This function is used to extract solution from a built tree.
    It is mainly used for MCTS, but also works for saved tree from step_beam.
    """
    question = full_tree_dict["question"] if "question" in full_tree_dict else full_tree_dict["query"]
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
    
    traces = search_all_traces(root, mode=mode)
    
    return traces


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="")
    args.add_argument("--output_file", type=str, default="")
    args.add_argument("--mode", type=str, default="all")
    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cnt, total = 0, 0
    solutions = []
  
    full_tree_dicts = []
    for dir in os.listdir(args.data_dir):
        if not dir.endswith(".jsonl"): continue
        if "extra" in dir: continue
        with open(args.data_dir + "/" + dir, "r") as f:
            for line in f:
                try:
                    full_tree_dict = json.loads(line)
                    full_tree_dicts.append(full_tree_dict)
                except:
                    pass
        
    
    def get_one_solution(full_tree_dict):
        solution = extra_solution_dict(
                    full_tree_dict,
                    mode=args.mode,
                )
        return solution
        

    def save_batch_to_jsonl(solutions, file_path):
        with open(file_path, "a") as f:
            for solution in solutions:
                json.dump(solution, f)
                f.write("\n")
    batch_size = 5000
    output_file = args.output_file

    #test = get_one_solution(full_tree_dicts[0])
    with ProcessPool(max_workers=os.cpu_count() - 8) as pool:
        executor = partial(get_one_solution)
        
        for i in range(0, len(full_tree_dicts), batch_size):
            batch = full_tree_dicts[i:i + batch_size]
            future = pool.map(executor, batch, timeout=60)
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