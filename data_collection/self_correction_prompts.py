"""
Self-Correction Prompt Templates for Generating Training Data

These prompts are designed to generate multi-turn self-correction trajectories
for fine-tuning models on the interactive code agent workflow.

Input: Math problem + correct answer
Output: Multi-turn trajectory with error -> correction -> correct answer
"""

# System prompt for the generator model (e.g., Grok, GPT-4)
SYSTEM_PROMPT = """You are a math tutor generating training data for teaching AI models to self-correct.

Your task: Given a math problem and its correct answer, generate a realistic multi-turn solution trajectory where:
1. First attempt has a specific type of error
2. The model recognizes the error from the output
3. The model corrects itself and arrives at the correct answer

Format requirements:
- Use ```python and ``` for code blocks
- Use ```output and ``` for execution results
- End with \\boxed{answer} format
- Make errors realistic and corrections natural"""


# Template 1: Logic Error -> Self Correction
LOGIC_ERROR_TEMPLATE = """Generate a math solution with the following pattern:

Problem: {question}
Correct Answer: {answer}

Requirements:
1. First, write Python code that has a LOGIC ERROR (wrong formula, misunderstood problem, calculation mistake)
2. Show the wrong output
3. Then realize the mistake by re-reading the problem
4. Write corrected code
5. Show correct output
6. Give final answer in \\boxed{{}}

Example format:

Let me solve this step by step using Python.

```python
# First attempt - with a logic error
# [code with wrong logic]
print(wrong_result)
```
```output
[wrong number]
```

Wait, let me re-read the problem. I made a mistake - [explain the error]. Let me fix this:

```python
# Corrected approach
# [fixed code]
print(correct_result)
```
```output
{answer}
```

The answer is \\boxed{{{answer}}}.

---
Now generate for the given problem. Make the logic error realistic (e.g., misreading "per week" as "per day", forgetting a step, using wrong operation)."""


# Template 2: NameError -> Self Correction  
NAME_ERROR_TEMPLATE = """Generate a math solution with the following pattern:

Problem: {question}
Correct Answer: {answer}

Requirements:
1. First, write Python code that causes a NameError (using undefined variable)
2. Show the error message
3. Recognize the error and fix it
4. Show correct output
5. Give final answer in \\boxed{{}}

Example format:

Let me solve this step by step using Python.

```python
# First attempt - uses undefined variable
result = x + y * 2  # x is not defined
print(result)
```
```output
NameError: name 'x' is not defined
```

I see the error - I used variable 'x' before defining it. Let me fix this:

```python
# Define all variables first
x = [correct value]
y = [correct value]
result = x + y * 2
print(result)
```
```output
{answer}
```

The answer is \\boxed{{{answer}}}.

---
Now generate for the given problem. Make sure the NameError is realistic (common mistake when coding quickly)."""


# Template 3: Syntax Error -> Self Correction
SYNTAX_ERROR_TEMPLATE = """Generate a math solution with the following pattern:

Problem: {question}
Correct Answer: {answer}

Requirements:
1. First, write Python code with a syntax error (missing parenthesis, colon, etc.)
2. Show the syntax error message
3. Fix the syntax
4. Show correct output
5. Give final answer in \\boxed{{}}

Example format:

Let me solve this step by step using Python.

```python
# First attempt - syntax error
for i in range(10)  # missing colon
    total += i
print(total)
```
```output
SyntaxError: expected ':'
```

I see there's a syntax error - missing colon after the for statement. Let me fix it:

```python
total = 0
for i in range(10):
    total += i
print(total)
```
```output
{answer}
```

The answer is \\boxed{{{answer}}}.

---
Now generate for the given problem."""


# Template 4: TypeError -> Self Correction
TYPE_ERROR_TEMPLATE = """Generate a math solution with the following pattern:

Problem: {question}
Correct Answer: {answer}

Requirements:
1. First, write Python code that causes a TypeError
2. Show the error message
3. Fix the type issue
4. Show correct output
5. Give final answer in \\boxed{{}}

Example format:

Let me solve this step by step using Python.

```python
# First attempt - type error
numbers = "123"
result = numbers * 2.5  # can't multiply string by float
print(result)
```
```output
TypeError: can't multiply sequence by non-int of type 'float'
```

I see the error - I'm trying to multiply a string by a float. Let me convert to the right type:

```python
numbers = 123
result = numbers * 2.5
print(result)
```
```output
{answer}
```

The answer is \\boxed{{{answer}}}.

---
Now generate for the given problem."""


# Template 5: Wrong Output -> Re-check -> Correction
WRONG_OUTPUT_TEMPLATE = """Generate a math solution with the following pattern:

Problem: {question}
Correct Answer: {answer}

Requirements:
1. First, write Python code that runs but gives wrong answer
2. Show the wrong output
3. Notice the output doesn't make sense, re-check the logic
4. Write corrected code
5. Show correct output
6. Give final answer in \\boxed{{}}

Example format:

Let me solve this step by step using Python.

```python
# First attempt
[code that gives wrong answer]
print(result)
```
```output
[wrong number, e.g., negative when should be positive, too large/small]
```

Hmm, that doesn't seem right. [Explain why the output looks wrong - e.g., "a negative number of apples doesn't make sense"]. Let me reconsider the problem:

[Re-analyze the problem]

```python
# Corrected approach
[fixed code]
print(result)
```
```output
{answer}
```

The answer is \\boxed{{{answer}}}.

---
Now generate for the given problem. The wrong output should be obviously incorrect (negative count, impossibly large number, etc.)."""


# Template 6: Multi-step Correction (2 errors)
MULTI_STEP_TEMPLATE = """Generate a math solution with the following pattern:

Problem: {question}
Correct Answer: {answer}

Requirements:
1. First attempt: has an error (runtime or logic)
2. Second attempt: fixes first error but has another issue
3. Third attempt: finally correct
4. Give final answer in \\boxed{{}}

Example format:

Let me solve this step by step using Python.

```python
# First attempt
[code with error 1]
```
```output
[Error or wrong result]
```

[Recognize error 1 and attempt to fix]

```python
# Second attempt - fixed error 1 but has error 2
[code with different error]
```
```output
[Different error or still wrong]
```

[Recognize error 2]

```python
# Third attempt - all correct
[correct code]
print(result)
```
```output
{answer}
```

The answer is \\boxed{{{answer}}}.

---
Now generate for the given problem. Make errors progressively more subtle."""


# All templates
TEMPLATES = {
    'logic_error': LOGIC_ERROR_TEMPLATE,
    'name_error': NAME_ERROR_TEMPLATE,
    'syntax_error': SYNTAX_ERROR_TEMPLATE,
    'type_error': TYPE_ERROR_TEMPLATE,
    'wrong_output': WRONG_OUTPUT_TEMPLATE,
    'multi_step': MULTI_STEP_TEMPLATE,
}


def get_prompt(template_name: str, question: str, answer: str) -> str:
    """Get formatted prompt for a specific error type."""
    template = TEMPLATES.get(template_name, LOGIC_ERROR_TEMPLATE)
    return template.format(question=question, answer=answer)


def get_all_prompts(question: str, answer: str) -> dict:
    """Get all prompt variants for a question."""
    return {
        name: get_prompt(name, question, answer)
        for name in TEMPLATES
    }


# API call function for Grok/OpenAI
def generate_correction_trajectory(
    question: str,
    answer: str,
    error_type: str = 'logic_error',
    api_key: str = None,
    model: str = "grok-3-fast",
    base_url: str = "https://api.x.ai/v1"
):
    """
    Call API to generate a self-correction trajectory.
    
    Args:
        question: Math problem
        answer: Correct answer
        error_type: Type of error to simulate
        api_key: API key
        model: Model name
        base_url: API base URL
    
    Returns:
        Generated trajectory string
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    prompt = get_prompt(error_type, question, answer)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    
    return response.choices[0].message.content


if __name__ == "__main__":
    # Test
    test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    test_answer = "18"
    
    print("=== Logic Error Template ===")
    print(get_prompt('logic_error', test_question, test_answer))
    
    print("\n\n=== Name Error Template ===")
    print(get_prompt('name_error', test_question, test_answer))

