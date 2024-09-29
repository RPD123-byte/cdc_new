helpful_response_generator_prompt_template = """
    You are an AI assistant that provides helpful, detailed answers to user questions based on the given context.
"""

search_query_generator_prompt_template = """
You are a search and retrieve agent. Your task is to take a user question and format it into a structured search query that adheres to the provided Pydantic schemas.

### **Valid Columns for Each Collection:** 
{column_names}

### **Search Criteria Structure:**
The search criteria should be a single Python dictionary in the following format:

search_criteria = {{
    'column_1': 'Value or condition',
    'column_2': 'Value or condition',
    ...
    'columns': ['Column 1', 'Column 2', ...]  # or 'columns': 'all'
}}

- **Keys (excluding 'columns')**: Represent the search conditions where each key corresponds to a column name across collections.
- **Values**: The value to search for or a condition string for numerical columns.
- **'columns' Key**: Specifies which columns to retrieve. It can be set to 'all' to retrieve all columns or provide a list of specific column names.

### **Example:**

For the question:

"Restaurants in Amsterdam near longitude 50 to 60"

A valid search query would look like this:

```python
search_criteria = {{
    'location': 'Amsterdam',
    'category': 'restaurant',
    'lng': '50<x<60',
    'columns': 'all'
}}

Try to recommend places with an international phone number and a website, don't put any values in for those columns.

These are previous mistakes you made and this is your feedback:
{feedback}

"""

def get_search_query_generator_guided_json(column_names):
    """
    Generates a JSON schema for a flat search criteria dictionary.

    :param column_names: A list of valid column names.
    :return: A dictionary representing the JSON schema.
    """
    return {
        "type": "object",
        "properties": {
            **{
                col: {
                    "type": "string",
                    "description": f"Filter value or condition for column '{col}'. For numerical columns, use condition strings like '50<x<53'."
                }
                for col in column_names
            },
            "columns": {
                "oneOf": [
                    {
                        "type": "string",
                        "enum": ["all"],
                        "description": "Retrieve all columns."
                    },
                    {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": column_names,
                            "description": "A specific column to retrieve."
                        },
                        "minItems": 1,
                        "description": "List of specific columns to retrieve."
                    }
                ],
                "description": "Specify which columns to retrieve. Use 'all' to retrieve all columns or provide a list of specific column names."
            }
        },
        "required": ["columns"],
        "additionalProperties": False,
        "description": "A flat dictionary containing search conditions and a 'columns' key to specify retrieved columns."
    }
