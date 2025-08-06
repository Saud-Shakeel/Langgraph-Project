from langchain.tools import tool

# Tools
@tool("get_ticket_price", description="Return a mock ticket price for a DESTINATION CITY.")
def get_ticket_price(destination_city: str) -> str:
    prices = {"dubai": "$456.9", "islamabad": "$100.0", "tokyo": "$561.2", "mumbai": "$200.2"}
    return prices.get(destination_city.lower(), "ticket price not available")

@tool("get_stock_price", description="Return a mock stock price for the given COMPANY NAME.")
def get_stock_price(company_name: str) -> str:
    stocks = {"Microsoft": "250.2", "Apple": "350.5", "Google": "500.0", "Amazon": "400.7"}
    return stocks.get(company_name, "stock price not available")

tool_list = [get_stock_price, get_ticket_price]