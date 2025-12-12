"""
Generate SFT Verifier Training Data
This script creates 500 examples for training a verifier model to classify solutions as CORRECT/INCORRECT/UNCLEAR.
"""

import json
import csv
import random
import os

# Set seed for reproducibility
random.seed(42)

def generate_correct_example(idx):
    """Generate examples with correct solutions"""
    problem_types = [
        # Type 1: Simple multiplication
        lambda i: {
            'question': f'A store sells notebooks for $3 each. If Sarah buys {5+i%10} notebooks, how much does she spend?',
            'ground_truth': str(3 * (5 + i % 10)),
            'solution': f'''To solve this problem, I need to multiply the price per notebook by the number of notebooks.

Step-by-step reasoning:
1. Price per notebook: $3
2. Number of notebooks: {5+i%10}
3. Total cost: $3 × {5+i%10} = ${3*(5+i%10)}

Let me write code to compute this:
```python
price_per_notebook = 3
num_notebooks = {5+i%10}
total_cost = price_per_notebook * num_notebooks
final_answer = total_cost
print(final_answer)
```

The code outputs {3*(5+i%10)}, so the answer is \\boxed{{{3*(5+i%10)}}}''',
            'verdict': 'CORRECT',
            'reasoning': 'The solution correctly identifies the multiplication needed, performs the arithmetic accurately, and the code execution matches the manual calculation. The final answer is properly formatted and matches the ground truth.'
        },
        # Type 2: Distance = speed × time
        lambda i: {
            'question': f'If a train travels at {40+i%20} mph for {2+i%3} hours, how far does it travel?',
            'ground_truth': str((40+i%20) * (2+i%3)),
            'solution': f'''To find the distance traveled, I use the formula: distance = speed × time.

Given:
- Speed: {40+i%20} mph
- Time: {2+i%3} hours

Calculation:
Distance = {40+i%20} × {2+i%3} = {(40+i%20)*(2+i%3)} miles

Let me verify with code:
```python
speed = {40+i%20}
time = {2+i%3}
distance = speed * time
final_answer = distance
print(final_answer)
```

The code outputs {(40+i%20)*(2+i%3)}, confirming the answer is \\boxed{{{(40+i%20)*(2+i%3)}}}''',
            'verdict': 'CORRECT',
            'reasoning': 'The solution applies the correct formula (distance = speed × time), performs accurate calculations, and the code verification confirms the result. The answer format is correct.'
        },
        # Type 3: Percentage calculation
        lambda i: {
            'question': f'A shirt costs ${20+i%30}. If there is a {10+i%20}% discount, what is the final price?',
            'ground_truth': str(round((20+i%30) * (1 - (10+i%20)/100), 2)),
            'solution': f'''To calculate the final price after discount:

Step-by-step reasoning:
1. Original price: ${20+i%30}
2. Discount percentage: {10+i%20}%
3. Discount amount: ${20+i%30} × {10+i%20}/100 = ${round((20+i%30)*(10+i%20)/100, 2)}
4. Final price: ${20+i%30} - ${round((20+i%30)*(10+i%20)/100, 2)} = ${round((20+i%30) * (1 - (10+i%20)/100), 2)}

Let me verify with code:
```python
original_price = {20+i%30}
discount_percent = {10+i%20}
discount_amount = original_price * discount_percent / 100
final_price = original_price - discount_amount
final_answer = round(final_price, 2)
print(final_answer)
```

The code outputs {round((20+i%30) * (1 - (10+i%20)/100), 2)}, so the answer is \\boxed{{{round((20+i%30) * (1 - (10+i%20)/100), 2)}}}''',
            'verdict': 'CORRECT',
            'reasoning': 'The solution correctly calculates the discount amount and subtracts it from the original price. The code implementation matches the manual calculation, and the final answer is accurate.'
        },
        # Type 4: Average calculation
        lambda i: {
            'question': f'What is the average of {10+i%5}, {15+i%7}, and {20+i%8}?',
            'ground_truth': str(round(((10+i%5) + (15+i%7) + (20+i%8)) / 3, 2)),
            'solution': f'''To find the average, I sum the numbers and divide by the count.

Step-by-step reasoning:
1. Numbers: {10+i%5}, {15+i%7}, {20+i%8}
2. Sum: {10+i%5} + {15+i%7} + {20+i%8} = {(10+i%5) + (15+i%7) + (20+i%8)}
3. Count: 3
4. Average: {(10+i%5) + (15+i%7) + (20+i%8)} / 3 = {round(((10+i%5) + (15+i%7) + (20+i%8)) / 3, 2)}

Let me verify with code:
```python
numbers = [{10+i%5}, {15+i%7}, {20+i%8}]
average = sum(numbers) / len(numbers)
final_answer = round(average, 2)
print(final_answer)
```

The code outputs {round(((10+i%5) + (15+i%7) + (20+i%8)) / 3, 2)}, so the answer is \\boxed{{{round(((10+i%5) + (15+i%7) + (20+i%8)) / 3, 2)}}}''',
            'verdict': 'CORRECT',
            'reasoning': 'The solution correctly applies the average formula (sum/count), performs accurate arithmetic, and the code verification confirms the result.'
        }
    ]
    
    problem_func = random.choice(problem_types)
    return problem_func(idx)

def generate_arithmetic_error_example(idx):
    """Generate examples with arithmetic errors"""
    error_val = random.randint(1, 3)
    problem_types = [
        # Type 1: Subtraction error
        lambda i, err: {
            'question': f'John has {20+i%15} apples. He gives away {5+i%8} apples. How many apples does he have left?',
            'ground_truth': str((20+i%15) - (5+i%8)),
            'solution': f'''To solve this problem, I need to subtract the apples given away from the total.

Step-by-step reasoning:
1. Initial apples: {20+i%15}
2. Apples given away: {5+i%8}
3. Remaining apples: {20+i%15} - {5+i%8} = {(20+i%15)-(5+i%8)+err}

Let me write code to compute this:
```python
initial_apples = {20+i%15}
given_away = {5+i%8}
remaining = initial_apples - given_away
final_answer = remaining
print(final_answer)
```

The code outputs {(20+i%15)-(5+i%8)}, but I calculated {(20+i%15)-(5+i%8)+err} manually. So the answer is \\boxed{{{(20+i%15)-(5+i%8)+err}}}''',
            'verdict': 'INCORRECT',
            'reasoning': f'Arithmetic error detected. The manual calculation states {(20+i%15)-(5+i%8)+err}, but the code correctly computes {(20+i%15)-(5+i%8)}. The subtraction was performed incorrectly in the reasoning steps. The final answer does not match the ground truth.'
        },
        # Type 2: Addition error
        lambda i, err: {
            'question': f'A basket has {12+i%10} oranges and {8+i%7} apples. How many fruits are there in total?',
            'ground_truth': str((12+i%10) + (8+i%7)),
            'solution': f'''To find the total number of fruits, I need to add oranges and apples.

Step-by-step reasoning:
1. Oranges: {12+i%10}
2. Apples: {8+i%7}
3. Total fruits: {12+i%10} + {8+i%7} = {(12+i%10)+(8+i%7)-err}

Let me write code to compute this:
```python
oranges = {12+i%10}
apples = {8+i%7}
total_fruits = oranges + apples
final_answer = total_fruits
print(final_answer)
```

The code outputs {(12+i%10)+(8+i%7)}, so the answer is \\boxed{{{(12+i%10)+(8+i%7)-err}}}''',
            'verdict': 'INCORRECT',
            'reasoning': f'Arithmetic error detected. The solution states the sum is {(12+i%10)+(8+i%7)-err}, but the code correctly computes {(12+i%10)+(8+i%7)}. There is a calculation mistake in step 3.'
        },
        # Type 3: Multiplication error
        lambda i, err: {
            'question': f'If one box contains {6+i%5} pencils, how many pencils are in {4+i%3} boxes?',
            'ground_truth': str((6+i%5) * (4+i%3)),
            'solution': f'''To find the total pencils, I multiply pencils per box by number of boxes.

Step-by-step reasoning:
1. Pencils per box: {6+i%5}
2. Number of boxes: {4+i%3}
3. Total pencils: {6+i%5} × {4+i%3} = {(6+i%5)*(4+i%3)+err*2}

Let me write code to compute this:
```python
pencils_per_box = {6+i%5}
num_boxes = {4+i%3}
total_pencils = pencils_per_box * num_boxes
final_answer = total_pencils
print(final_answer)
```

The code outputs {(6+i%5)*(4+i%3)}, confirming the answer is \\boxed{{{(6+i%5)*(4+i%3)+err*2}}}''',
            'verdict': 'INCORRECT',
            'reasoning': f'Arithmetic error detected. The manual calculation in step 3 shows {(6+i%5)*(4+i%3)+err*2}, but the code correctly computes {(6+i%5)*(4+i%3)}. The multiplication was performed incorrectly. The final boxed answer does not match the code output or ground truth.'
        }
    ]
    
    problem_func = random.choice(problem_types)
    return problem_func(idx, error_val)

def generate_logic_error_example(idx):
    """Generate examples with logical/formula errors"""
    problem_types = [
        # Type 1: Wrong perimeter formula
        lambda i: {
            'question': f'A rectangle has length {10+i%8} cm and width {6+i%5} cm. What is its perimeter?',
            'ground_truth': str(2 * ((10+i%8) + (6+i%5))),
            'solution': f'''To find the perimeter of a rectangle, I need to add all four sides.

Step-by-step reasoning:
1. Length: {10+i%8} cm
2. Width: {6+i%5} cm
3. Perimeter = length + width = {10+i%8} + {6+i%5} = {(10+i%8)+(6+i%5)} cm

Let me write code to compute this:
```python
length = {10+i%8}
width = {6+i%5}
perimeter = length + width
final_answer = perimeter
print(final_answer)
```

The code outputs {(10+i%8)+(6+i%5)}, so the answer is \\boxed{{{(10+i%8)+(6+i%5)}}}''',
            'verdict': 'INCORRECT',
            'reasoning': f'Logic error detected. The solution uses the formula perimeter = length + width, but the correct formula for a rectangle is perimeter = 2 × (length + width). The code implements the incorrect formula. The correct answer should be {2*((10+i%8)+(6+i%5))}, not {(10+i%8)+(6+i%5)}.'
        },
        # Type 2: Wrong area formula
        lambda i: {
            'question': f'A triangle has base {8+i%6} cm and height {12+i%7} cm. What is its area?',
            'ground_truth': str(round(0.5 * (8+i%6) * (12+i%7), 2)),
            'solution': f'''To find the area of a triangle, I multiply base and height.

Step-by-step reasoning:
1. Base: {8+i%6} cm
2. Height: {12+i%7} cm
3. Area = base × height = {8+i%6} × {12+i%7} = {(8+i%6)*(12+i%7)} cm²

Let me write code to compute this:
```python
base = {8+i%6}
height = {12+i%7}
area = base * height
final_answer = area
print(final_answer)
```

The code outputs {(8+i%6)*(12+i%7)}, so the answer is \\boxed{{{(8+i%6)*(12+i%7)}}}''',
            'verdict': 'INCORRECT',
            'reasoning': f'Logic error detected. The solution uses area = base × height, but the correct formula for a triangle is area = (1/2) × base × height. The code implements the incorrect formula, missing the 1/2 factor. The correct answer should be {round(0.5 * (8+i%6) * (12+i%7), 2)}, not {(8+i%6)*(12+i%7)}.'
        },
        # Type 3: Wrong average calculation
        lambda i: {
            'question': f'Tom scored {85+i%10}, {90+i%8}, and {78+i%12} on three tests. What is his average score?',
            'ground_truth': str(round(((85+i%10) + (90+i%8) + (78+i%12)) / 3, 2)),
            'solution': f'''To find the average, I sum all scores.

Step-by-step reasoning:
1. Test 1: {85+i%10}
2. Test 2: {90+i%8}
3. Test 3: {78+i%12}
4. Average = {85+i%10} + {90+i%8} + {78+i%12} = {(85+i%10) + (90+i%8) + (78+i%12)}

Let me write code to compute this:
```python
scores = [{85+i%10}, {90+i%8}, {78+i%12}]
average = sum(scores)
final_answer = average
print(final_answer)
```

The code outputs {(85+i%10) + (90+i%8) + (78+i%12)}, so the answer is \\boxed{{{(85+i%10) + (90+i%8) + (78+i%12)}}}''',
            'verdict': 'INCORRECT',
            'reasoning': f'Logic error detected. The solution calculates the sum but forgets to divide by the number of tests. The correct formula for average is sum/count. The code implements the incorrect formula. The correct answer should be {round(((85+i%10) + (90+i%8) + (78+i%12)) / 3, 2)}, not {(85+i%10) + (90+i%8) + (78+i%12)}.'
        }
    ]
    
    problem_func = random.choice(problem_types)
    return problem_func(idx)

def generate_code_error_example(idx):
    """Generate examples with code execution errors"""
    problem_types = [
        # Type 1: Off-by-one error in range
        lambda i: {
            'question': f'Calculate the sum of numbers from 1 to {5+i%10}.',
            'ground_truth': str(sum(range(1, 6+i%10))),
            'solution': f'''To find the sum of numbers from 1 to {5+i%10}, I can use iteration.

Step-by-step reasoning:
1. Numbers: 1, 2, 3, ..., {5+i%10}
2. Sum = 1 + 2 + 3 + ... + {5+i%10}

Let me write code to compute this:
```python
n = {5+i%10}
total = sum(range(1, n))
final_answer = total
print(final_answer)
```

The code outputs {sum(range(1, 5+i%10))}, so the answer is \\boxed{{{sum(range(1, 5+i%10))}}}''',
            'verdict': 'INCORRECT',
            'reasoning': f'Code execution error detected. The code uses range(1, n) which excludes n itself. The correct range should be range(1, n+1) to include {5+i%10}. The computed answer {sum(range(1, 5+i%10))} is incorrect; the correct answer is {sum(range(1, 6+i%10))}.'
        },
        # Type 2: Integer division error
        lambda i: {
            'question': f'{20+i%15} cookies are shared equally among {3+i%2} children. How many cookies does each child get?',
            'ground_truth': str(round((20+i%15) / (3+i%2), 2)),
            'solution': f'''To find cookies per child, I divide total cookies by number of children.

Step-by-step reasoning:
1. Total cookies: {20+i%15}
2. Number of children: {3+i%2}
3. Cookies per child: {20+i%15} / {3+i%2}

Let me write code to compute this:
```python
total_cookies = {20+i%15}
num_children = {3+i%2}
cookies_per_child = total_cookies // num_children
final_answer = cookies_per_child
print(final_answer)
```

The code outputs {(20+i%15) // (3+i%2)}, so the answer is \\boxed{{{(20+i%15) // (3+i%2)}}}''',
            'verdict': 'INCORRECT',
            'reasoning': f'Code execution error detected. The code uses integer division (//) which truncates the result, but the problem asks for the actual division result. The code should use regular division (/) to get {round((20+i%15) / (3+i%2), 2)}, not {(20+i%15) // (3+i%2)}. This is a semantic error in choosing the wrong operator.'
        }
    ]
    
    problem_func = random.choice(problem_types)
    return problem_func(idx)

def generate_unclear_example(idx):
    """Generate examples with unclear or ambiguous solutions"""
    problem_types = [
        # Type 1: Self-contradictory reasoning
        lambda i: {
            'question': f'A bag contains {10+i%15} red balls and some blue balls. If there are {25+i%20} balls in total, how many blue balls are there?',
            'ground_truth': str((25+i%20) - (10+i%15)),
            'solution': f'''To solve this problem, I need to find the number of blue balls.

Step-by-step reasoning:
1. Total balls: {25+i%20}
2. Red balls: {10+i%15}
3. Blue balls: {25+i%20} - {10+i%15} = {(25+i%20)-(10+i%15)}

Let me write code to compute this:
```python
total_balls = {25+i%20}
red_balls = {10+i%15}
blue_balls = total_balls - red_balls
final_answer = blue_balls
print(final_answer)
```

The code outputs {(25+i%20)-(10+i%15)}, so the answer is \\boxed{{{(25+i%20)-(10+i%15)}}}

Wait, let me verify: {25+i%20} - {10+i%15} = {(25+i%20)-(10+i%15)+random.choice([-1,0,1])}? I'm getting slightly different values when I recalculate.''',
            'verdict': 'UNCLEAR',
            'reasoning': 'The solution shows uncertainty and inconsistency. The solver expresses doubt about the calculation and mentions getting different values. While the code output appears correct, the reasoning contains self-contradiction and lacks confidence. This requires clarification or re-verification.'
        },
        # Type 2: Missing steps or incomplete reasoning
        lambda i: {
            'question': f'If {3+i%4} workers can complete a job in {12+i%8} days, how many days will it take {6+i%5} workers?',
            'ground_truth': str(round((3+i%4) * (12+i%8) / (6+i%5), 2)),
            'solution': f'''To solve this work rate problem, I need to consider the relationship between workers and time.

Step-by-step reasoning:
1. Original: {3+i%4} workers, {12+i%8} days
2. New: {6+i%5} workers, ? days
3. More workers means less time...

Let me write code to compute this:
```python
workers1 = {3+i%4}
days1 = {12+i%8}
workers2 = {6+i%5}
# Not sure about the exact formula here
days2 = (workers1 * days1) / workers2
final_answer = round(days2, 2)
print(final_answer)
```

The code outputs {round((3+i%4) * (12+i%8) / (6+i%5), 2)}, so the answer is \\boxed{{{round((3+i%4) * (12+i%8) / (6+i%5), 2)}}}

But I'm not entirely confident about this formula.''',
            'verdict': 'UNCLEAR',
            'reasoning': 'The solution lacks confidence and explicitly states uncertainty about the formula used. While the code produces a result, the reasoning is incomplete and the solver expresses doubt. The answer may be correct, but the solution process needs clarification and stronger justification.'
        },
        # Type 3: Ambiguous formatting
        lambda i: {
            'question': f'What is {15+i%10} divided by {3+i%2}?',
            'ground_truth': str(round((15+i%10) / (3+i%2), 2)),
            'solution': f'''To solve this division problem:

Step-by-step reasoning:
1. Dividend: {15+i%10}
2. Divisor: {3+i%2}
3. Result: {15+i%10} / {3+i%2} = {round((15+i%10) / (3+i%2), 2)}

Let me write code to compute this:
```python
dividend = {15+i%10}
divisor = {3+i%2}
result = dividend / divisor
final_answer = result
print(final_answer)
```

The code outputs {(15+i%10) / (3+i%2)}, so the answer is {(15+i%10) / (3+i%2)} or maybe {round((15+i%10) / (3+i%2), 2)}? Should I box the exact decimal or rounded version?''',
            'verdict': 'UNCLEAR',
            'reasoning': 'The solution is ambiguous about the final answer format. The solver is uncertain whether to provide the exact decimal or rounded version, and the boxed answer is missing. While the calculation appears correct, the presentation lacks clarity and decisiveness.'
        }
    ]
    
    problem_func = random.choice(problem_types)
    return problem_func(idx)

def generate_verifier_examples(n=500):
    """Generate n verifier training examples with distribution:
    - 60% correct
    - 20% arithmetic errors
    - 10% logic errors
    - 5% code errors
    - 5% unclear
    """
    examples = []
    
    # Calculate counts for each category
    categories = [
        ('correct', int(n * 0.60), generate_correct_example),
        ('arithmetic_error', int(n * 0.20), generate_arithmetic_error_example),
        ('logic_error', int(n * 0.10), generate_logic_error_example),
        ('code_error', int(n * 0.05), generate_code_error_example),
        ('unclear', int(n * 0.05), generate_unclear_example)
    ]
    
    # Adjust last category to ensure exactly n examples
    total_allocated = sum(count for _, count, _ in categories)
    if total_allocated < n:
        categories[-1] = (categories[-1][0], categories[-1][1] + (n - total_allocated), categories[-1][2])
    
    idx = 1
    for category_name, count, generator_func in categories:
        for _ in range(count):
            example = generator_func(idx)
            example['index'] = idx
            example['category'] = category_name
            examples.append(example)
            idx += 1
    
    # Shuffle to mix categories
    random.shuffle(examples)
    
    # Reassign indices after shuffling
    for i, example in enumerate(examples, 1):
        example['index'] = i
    
    return examples

def save_to_jsonl(examples, filepath):
    """Save examples to JSONL format"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

def save_to_csv(examples, filepath):
    """Save examples to CSV format"""
    fieldnames = ['index', 'question', 'ground_truth', 'solution', 'verdict', 'verification_reasoning', 'category']
    
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for example in examples:
            # Normalize field names
            row = {
                'index': example['index'],
                'question': example['question'],
                'ground_truth': example['ground_truth'],
                'solution': example['solution'],
                'verdict': example['verdict'],
                'verification_reasoning': example.get('reasoning', example.get('verification_reasoning', '')),
                'category': example['category']
            }
            writer.writerow(row)

def main():
    # Create output directory
    output_dir = os.path.join('data', 'sft_verifier_training')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate examples
    print("Generating 500 verifier training examples...")
    examples = generate_verifier_examples(500)
    
    # Save to files
    jsonl_path = os.path.join(output_dir, 'sft_verifier_training_data.jsonl')
    csv_path = os.path.join(output_dir, 'sft_verifier_training_data.csv')
    
    print(f"Saving to JSONL: {jsonl_path}")
    save_to_jsonl(examples, jsonl_path)
    
    print(f"Saving to CSV: {csv_path}")
    save_to_csv(examples, csv_path)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total examples: {len(examples)}")
    
    verdict_counts = {}
    category_counts = {}
    for example in examples:
        verdict = example['verdict']
        category = example['category']
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print("\nVerdict Distribution:")
    for verdict, count in sorted(verdict_counts.items()):
        print(f"  {verdict}: {count} ({count/len(examples)*100:.1f}%)")
    
    print("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} ({count/len(examples)*100:.1f}%)")
    
    print("\n=== Sample Examples ===")
    for verdict in ['CORRECT', 'INCORRECT', 'UNCLEAR']:
        sample = next((ex for ex in examples if ex['verdict'] == verdict), None)
        if sample:
            print(f"\n{verdict} Example (Index {sample['index']}):")
            print(f"Question: {sample['question'][:100]}...")
            print(f"Ground Truth: {sample['ground_truth']}")
            print(f"Verdict: {sample['verdict']}")
            reasoning = sample.get('reasoning', sample.get('verification_reasoning', 'N/A'))
            print(f"Reasoning: {reasoning[:150]}...")
    
    print("\n[SUCCESS] Dataset generation complete!")

if __name__ == '__main__':
    main()

