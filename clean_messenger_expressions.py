import pandas as pd
import re
import os

def clean_messenger_expressions(input_path, output_path):
    """
    Reads a CSV file, removes messenger-like expressions from the 'conversation' column
    for all rows, and saves the cleaned data to a new CSV file.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path to save the cleaned CSV file.
    """
    try:
        # Read the CSV file with specified encoding
        df = pd.read_csv(input_path, encoding='utf-8')

        # Define the regex pattern to remove
        # This pattern looks for one or more repetitions of ㅋ, ㅎ, ㅜ, ㅠ, or 키.
        messenger_pattern = re.compile(r'(ㅋ|ㅎ|ㅜ|ㅠ|키)+')

        # Check if 'conversation' column exists
        if 'conversation' in df.columns:
            # Apply the cleaning function to the entire 'conversation' column
            # Ensure the column is treated as string before applying regex
            df['conversation'] = df['conversation'].astype(str).str.replace(messenger_pattern, '', regex=True)
            
            # Save the cleaned DataFrame to a new CSV file
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"Cleaned data saved to {output_path}")
        else:
            print(f"Error: 'conversation' column not found in {input_path}")

    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Define file paths
    data_dir = os.path.join('Data', 'aiffel-dl-thon-dktc-online-15')
    input_csv_path = os.path.join(data_dir, 'aug_hub.csv')
    output_csv_path = os.path.join(data_dir, 'aug_hub_cleaned.csv')
    
    # Create the full absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_input_path = os.path.join(current_dir, input_csv_path)
    absolute_output_path = os.path.join(current_dir, output_csv_path)

    # Ensure the paths are normalized
    absolute_input_path = os.path.normpath(absolute_input_path)
    absolute_output_path = os.path.normpath(absolute_output_path)

    clean_messenger_expressions(absolute_input_path, absolute_output_path)
