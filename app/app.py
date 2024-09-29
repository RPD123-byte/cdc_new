from agent_graph.graph import create_graph, compile_workflow
import pandas as pd
import io
import pickle
from utils.helper_functions import load_config  # Ensure this is imported
import json
import re

load_config('config/config.yaml')  # Specify the correct path if different

server = 'openai'
model = 'gpt-4o'
model_endpoint = None



def load_collection_data_frames(filename='collection_data_frames.pkl'):
    """
    Loads the pickled DataFrame from the specified file.

    :param filename: Path to the pickle file.
    :return: Loaded Pandas DataFrame.
    """
    try:
        with open(filename, 'rb') as file:
            data_frames = pickle.load(file)
        print(f"Data successfully loaded from {filename}")
        return data_frames
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        raise e
    
location_data_frames = load_collection_data_frames("location_data_frames.pkl")

# Define collection_keys from the keys of the dictionary
column_names = ["location","category","place_id","place_name","reviews_text","address","international_phone_number","lat","lng","polarity","website"]


def parse_condition(cond_str, column, df):
    """
    Parses a condition string and returns a lambda function that applies the condition to a DataFrame column.

    Supported condition formats:
    - "x<value"
    - "x<=value"
    - "x>value"
    - "x>=value"
    - "x==value"
    - "x!=value"
    - "value1<x<value2"
    - "value1<=x<=value2"
    - "value1<x<=value2"
    - "value1<=x<value2"

    :param cond_str: Condition string (e.g., "0<x<50.4", "x<25").
    :param column: Column name to apply the condition on.
    :return: A Pandas Series representing the condition mask.
    """
    # Remove any spaces
    cond_str = cond_str.replace(" ", "")
    
    # Patterns for different condition types
    pattern_double = re.compile(r'^([<>]=?)(x)([<>]=?)([\d\.]+)$')
    pattern_between = re.compile(r'^([\d\.]+)([<>]=?)(x)([<>]=?)([\d\.]+)$')
    pattern_single = re.compile(r'^(x)([<>]=?)([\d\.]+)$')
    pattern_single_reverse = re.compile(r'^([\d\.]+)([<>]=?)(x)$')
    
    # Initialize mask as None
    mask = None
    
    # Check for between conditions like "0<x<50.4" or "10<=x<=20"
    match_between = re.match(r'^([\d\.]+)([<>]=?)(x)([<>]=?)([\d\.]+)$', cond_str)
    if match_between:
        val1, op1, _, op2, val2 = match_between.groups()
        val1 = float(val1)
        val2 = float(val2)
        if op1 in ['<', '<=']:
            condition1 = f"{column} {op1} {val2}"
        else:
            condition1 = f"{column} {op1} {val1}"
        
        if op2 in ['<', '<=']:
            condition2 = f"{column} {op2} {val2}"
        else:
            condition2 = f"{column} {op2} {val1}"
        
        # Apply both conditions
        mask = (df[column] > val1) if 'x>' in cond_str else (df[column] >= val1) if 'x>=' in cond_str else (df[column] < val2) if 'x<' in cond_str else (df[column] <= val2)
        # Alternatively, apply both conditions
        mask = (df[column] > val1) & (df[column] < val2)
        return mask
    
    # Check for single conditions like "x<25", "x>=10"
    match_single = re.match(r'^(x)([<>]=?)([\d\.]+)$', cond_str)
    if match_single:
        _, operator, value = match_single.groups()
        value = float(value)
        if operator == '<':
            mask = df[column] < value
        elif operator == '<=':
            mask = df[column] <= value
        elif operator == '>':
            mask = df[column] > value
        elif operator == '>=':
            mask = df[column] >= value
        elif operator == '==':
            mask = df[column] == value
        elif operator == '!=':
            mask = df[column] != value
        return mask
    
    # Check for reversed single conditions like "10<x"
    match_single_rev = re.match(r'^([\d\.]+)([<>]=?)(x)$', cond_str)
    if match_single_rev:
        value, operator, _ = match_single_rev.groups()
        value = float(value)
        if operator == '<':
            mask = df[column] > value
        elif operator == '<=':
            mask = df[column] >= value
        elif operator == '>':
            mask = df[column] < value
        elif operator == '>=':
            mask = df[column] <= value
        elif operator == '==':
            mask = df[column] == value
        elif operator == '!=':
            mask = df[column] != value
        return mask
    
    # If no pattern matches, raise an error
    raise ValueError(f"Unsupported condition format: {cond_str}")

def search_database(df, search_criteria):
    """
    Searches the DataFrame based on the provided search criteria.

    :param df: Pandas DataFrame to search.
    :param search_criteria: Dictionary containing search conditions and columns to return.
                            Example:
                            {
                                'location': 'Amsterdam',
                                'category': 'Accommodation',
                                'lat': '0<x<50.4',
                                'lng': 'x<25',
                                'columns': 'all'
                            }
    :return: Filtered Pandas DataFrame based on the search criteria.
    """
    mask = pd.Series([True] * len(df))
    
    # Iterate through each key in the search criteria
    for key, value in search_criteria.items():
        if key == 'columns':
            continue  # Handle columns later
        if key not in df.columns:
            print(f"Warning: Column '{key}' does not exist in the DataFrame and will be ignored.")
            continue
        
        if isinstance(value, str) and 'x' in value:
            try:
                condition_mask = parse_condition(value, key, df)
                mask &= condition_mask
            except ValueError as ve:
                print(f"Error parsing condition for column '{key}': {ve}")
                continue
        else:
            # Exact match
            mask &= df[key] == value
    
    # Apply the mask to filter the DataFrame
    filtered_df = df[mask]
    
    # Handle the 'columns' key
    columns_to_return = search_criteria.get('columns', 'all')
    if columns_to_return != 'all':
        if isinstance(columns_to_return, list):
            # Validate columns
            valid_columns = [col for col in columns_to_return if col in df.columns]
            invalid_columns = set(columns_to_return) - set(valid_columns)
            if invalid_columns:
                print(f"Warning: The following columns are invalid and will be ignored: {invalid_columns}")
            if valid_columns:
                filtered_df = filtered_df[valid_columns]
            else:
                print("Warning: No valid columns specified. Returning all columns.")
        else:
            print("Warning: 'columns' should be a list or 'all'. Returning all columns.")
    
    return filtered_df

iterations = 40
question = "What are the best restaurants in amsterdam"
print ("Creating graph and compiling workflow...")
graph = create_graph(server=server, model=model, model_endpoint=model_endpoint, question=question, column_names=column_names)
workflow = compile_workflow(graph)
print ("Graph and workflow created.")


if __name__ == "__main__":
    verbose = False
    query = "go"

    dict_inputs = {"query_question": query}
    thread = {"configurable": {"thread_id": "4"}}
    limit = {"recursion_limit": iterations}


    for event in workflow.stream(
        dict_inputs, limit
        ):
        if verbose:
            print("\nState Dictionary:", event)
        else:
            print("\n")

    with open("criteria.txt", "r") as file:
        search_criteria = json.load(file)

    result_search = search_database(location_data_frames, search_criteria)
    print(result_search)

    

    