import inspect
import re

import ollama
from dotenv import load_dotenv
from ollama import ChatResponse

load_dotenv()

from langsmith import traceable

MAX_ITERATIONS = 10
PROVIDER_OLLAMA = "ollama"
PROVIDER_ANTHROPIC = "anthropic"
MODEL_NAME_GEMMA = "gemma4:e4b-mlx"
MODEL_NAME_SONNET = "claude-sonnet-5"
MODEL_GEMMA = f"{PROVIDER_OLLAMA}:{MODEL_NAME_GEMMA}"
MODEL_CLAUDE = f"{PROVIDER_ANTHROPIC}:{MODEL_NAME_SONNET}"


# --- Tools (LangChain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """
    Gets product price in the catalog.
    Args:
        product: The name of the product.
    Returns:
        The price of the product.
    """
    print(f"    >> Executing get_product_price for {product}")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }
    return prices.get(product, 0.0)


@traceable(run_type="tool")
def get_discount_tier(price: float | str, tier: str) -> float:
    """
    Apply a discount tier to a price and return the discounted price.
    Args:
        price: The price of the product.
        tier: The name of the discount tier.
    Returns:
        The discounted price of the product.
    """
    if isinstance(price, str):
        price = float(price)

    print(f"    >> Executing get_discount_tier for {price} and {tier}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount_percent = discount_percentages.get(tier.lower(), 0.0)
    return round(price * (100.0 - discount_percent) / 100.0, 2)


@traceable(run_type="tool")
def ask_user_question(query: str, *addl_query) -> str:
    """
    Asks user a question and returns the answer.
    Args:
        query: The question to ask the human user.
    Returns:
        The user's answer.
    """
    query += ",".join(addl_query)
    print(f"    >> Executing ask_user_question for {query}")
    return input(f"    [Agent Asks]: {query}\n\t>> ")


# --- Helper Functions ---


def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_fn in tools_dict.items():
        orig_fn = getattr(tool_fn, "__wrapped__", tool_fn)
        signature = inspect.signature(orig_fn)
        docstring = inspect.getdoc(tool_fn) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n\n".join(descriptions)


@traceable(name="Raw ReAct Prompt", run_type="llm")
def ollama_chat_traced(messages, options) -> ChatResponse:
    return ollama.chat(model=MODEL_NAME_GEMMA, messages=messages, options=options)


# --- Prompt ---

tools = {
    "get_product_price": get_product_price,
    "get_discount_tier": get_discount_tier,
    "ask_user_question": ask_user_question,
}

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES - you must follow these EXACTLY:
1. NEVER assume or guess the price of a product. You MUST call get_product_price() to get the real price.
2. Only call get_discount_tier() AFTER you have received a price from get_product_price(). Pass the exact price. NEVER use a made-up number.
3. NEVER calculate discounts yourself using math. ALWAYS use the get_discount_tier() to do this.
4. If the user does not specify a discount tier, ask them which tier to use. NEVER assume the tier.

Answer the following questions as best you can. You have access to the following tools.

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

# --- Agent Loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(query: str):
    print(f"Query: {query}")

    action_re = re.compile(r"\nAction:\s*([^\n]+)")
    action_input_re = re.compile(r"\nAction Input:\s*([^\n]+)")
    final_answer_re = re.compile(r"\nFinal Answer:\s*([^\n]+)")

    prompt = react_prompt.format(question=query)
    scratchpad = ""
    final_answer = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        full_prompt = prompt + scratchpad

        response: ollama.ChatResponse = ollama_chat_traced(
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )

        output = response.message.content or ""
        print(f"    [LLM Output]: {output}")

        final_answer_match = final_answer_re.search(output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            break

        action_match = action_re.search(output)
        action_input_match = action_input_re.search(output)
        if not action_match or not action_input_match:
            raise RuntimeError("Next action or action input not found!")

        tool_name = action_match.group(1).strip()
        tool_args_raw = action_input_match.group(1).strip()

        print(f"    [Tool Selected] {tool_name} with args: {tool_args_raw}")

        # split on commas to separate selected args
        raw_args = [arg.strip() for arg in tool_args_raw.split(",")]

        # remove the '=' signs and any single or double quotes
        args = [arg.split("=", maxsplit=1)[-1].strip("'\"") for arg in raw_args]

        print(f"    [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            obs = f"Error: tool {tool_name} not found. Available tools: {list[str](tools.keys())}"
        else:
            obs = str(tools[tool_name](*args))

        print(f"    [Tool Output] {obs}")

        scratchpad += f"{output}\nObservation: {obs}\nThought:"

    if not final_answer:
        raise ValueError("Agent unable to answer!")

    return final_answer


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!\n")
    question = "What is the price for a laptop after applying my discount?"
    result = run_agent(question)
