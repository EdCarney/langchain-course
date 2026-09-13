import ollama
from dotenv import load_dotenv

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
def get_discount_tier(price: float, tier: str) -> float:
    """
    Apply a discount tier to a price and return the discounted price.
    Args:
        price: The price of the product.
        tier: The name of the discount tier.

    Returns:
        The discounted price of the product.
    """
    print(f"    >> Executing get_discount_tier for {price} and {tier}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount_percent = discount_percentages.get(tier, 0.0)
    return round(price * (100.0 - discount_percent) / 100.0, 2)


@traceable(run_type="tool")
def ask_user_question(query: str) -> str:
    """
    Asks user a question and returns the answer.
    Args:
        query: The question to ask the human user.

    Returns:
        The user's answer.
    """
    print(f"    >> Executing ask_user_question for {query}")
    return input(f"    [Agent Asks]: {query}\n\t>> ")


# --- Helper Functions ---

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Gets product price in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product.",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_discount_tier",
            "description": "Apply a discount tier to a price and return the discounted price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The price of the product.",
                    },
                    "tier": {
                        "type": "string",
                        "description": "The name of the discount tier.",
                    },
                },
                "required": ["price", "tier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_question",
            "description": "Asks user a question and returns the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to ask the human user.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages) -> ollama.ChatResponse:
    return ollama.chat(model=MODEL_NAME_GEMMA, tools=tools_for_llm, messages=messages)


# --- Agent Loop ---


@traceable(name="LangChain Agent Loop")
def run_agent(query: str):
    global ai_msg
    tool_dict = {
        "get_product_price": get_product_price,
        "get_discount_tier": get_discount_tier,
        "ask_user_question": ask_user_question,
    }
    print(f"Query: {query}")

    sys_msg = {
        "role": "system",
        "content": """
        You are a helpful shopping assistant. You have access to a product catalog tool and a discount tool. Note that all prices are in USD ($).
        
        STRICT RULES - you must follow these EXACTLY:
        1. NEVER assume or guess the price of a product. You MUST call get_product_price() to get the real price.
        2. Only call get_discount_tier() AFTER you have received a price from get_product_price(). Pass the exact price. NEVER use a made-up number.
        3. NEVER calculate discounts yourself using math. ALWAYS use the get_discount_tier() to do this.
        4. If the user does not specify a discount tier, ask them which tier to use. NEVER assume the tier.
        """,
    }
    human_msg = {"role": "user", "content": query}
    msgs: list[dict] = [sys_msg, human_msg]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        response: ollama.ChatResponse = ollama_chat_traced(msgs)
        ai_msg = response.message
        tool_calls = ai_msg.tool_calls

        if not tool_calls:
            break

        # limiting to only ONE tool call per iteration for simplicity
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"    [Tool Selected]: {tool_name}({tool_args})")

        tool_fn = tool_dict.get(tool_name)
        if not tool_fn:
            raise ValueError(f"Tool {tool_name} not found!")

        obs = tool_fn(**tool_args)
        print(f"    [Tool Result]: {obs}")

        msgs.append(ai_msg)
        msgs.append({"role": "tool", "content": str(obs), "tool_name": tool_name})

    if not ai_msg:
        raise ValueError("Agent unable to answer!")

    print(f"Final Answer: {ai_msg.content}")
    return ai_msg.content


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!\n")
    question = "What is the price for a laptop after applying the gold discount?"
    question = "What is the price for a headphones after applying my discount?"
    result = run_agent(question)
