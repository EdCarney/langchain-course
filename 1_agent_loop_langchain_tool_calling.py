from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gemma4:e4b-mlx"

# --- Tools (LangChain @tool decorator) ---

@tool
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

@tool
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
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount_percent = discount_percentages.get(tier, 0.0)
    return round(price * (1.0 - discount_percent), 2)

# --- Agent Loop ---


@traceable(name="LangChain Agent Loop")
def run_agent(query: str):
    pass

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!\n")
    result = run_agent("What is the price for a laptop after applying the gold discount?")