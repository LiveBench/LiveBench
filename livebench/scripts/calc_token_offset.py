'''
Calculates the average percent difference between predicted and actual input tokens from a log file.
'''
import re
import statistics

def calculate_average_token_offset(log_file_path):
    '''
    Reads a log file, extracts predicted and actual token counts, 
    and calculates the average percent difference.

    Args:
        log_file_path (str): The path to the input log file.

    Returns:
        float: The average percent difference, or None if no relevant lines are found.
    '''
    percent_differences = []
    # To store data for argmin and argmax
    min_offset_data = {'diff': float('inf'), 'predicted': None, 'actual': None, 'line': ''}
    max_offset_data = {'diff': float('-inf'), 'predicted': None, 'actual': None, 'line': ''}
    # Regex to find lines like "predicted input tokens: x, actual input tokens: y"
    # It captures the numbers for x and y.
    pattern = re.compile(r"predicted input tokens: (\d+), actual input tokens: (\d+)")

    try:
        with open(log_file_path, 'r') as f:
            for line_content in f:
                match = pattern.search(line_content)
                if match:
                    predicted_tokens = int(match.group(1))
                    actual_tokens = int(match.group(2))

                    if actual_tokens == 0:
                        # Avoid division by zero. 
                        # Decide how to handle this case (e.g., skip, count as 0% diff if predicted is also 0, etc.)
                        # For now, we'll skip if actual is 0 to prevent skewed results or errors.
                        print(f"Warning: Actual tokens is 0 for line: {line_content.strip()}. Skipping this entry.")
                        continue

                    percent_diff = ((actual_tokens - predicted_tokens) / actual_tokens) * 100
                    percent_differences.append(percent_diff)

                    # Update min and max offset data
                    if percent_diff < min_offset_data['diff']:
                        min_offset_data['diff'] = percent_diff
                        min_offset_data['predicted'] = predicted_tokens
                        min_offset_data['actual'] = actual_tokens
                        min_offset_data['line'] = line_content.strip()
                    
                    if percent_diff > max_offset_data['diff']:
                        max_offset_data['diff'] = percent_diff
                        max_offset_data['predicted'] = predicted_tokens
                        max_offset_data['actual'] = actual_tokens
                        max_offset_data['line'] = line_content.strip()
    
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    if not percent_differences:
        print("No relevant log entries found to calculate token offset.")
        return None

    average_offset = sum(percent_differences) / len(percent_differences)
    min_offset = min(percent_differences)
    max_offset = max(percent_differences)
    std_dev_offset = statistics.stdev(percent_differences) if len(percent_differences) > 1 else 0

    return {
        "average": average_offset,
        "min": min_offset,
        "max": max_offset,
        "std_dev": std_dev_offset,
        "argmin_data": min_offset_data,
        "argmax_data": max_offset_data
    }

if __name__ == "__main__":
    # Placeholder for the input log file path
    # Replace this with the actual path to your log file
    input_log_file = "/home/gabriel/livebench/LiveBench/livebench/agentic_code_runner/data/trajectories/CU8RFXVvtCKPnx2wPahv3b/b948e8c8a905f4270afc6de4c9e24fa337399c5656e0fabaf63a19e6c1860fd1/b948e8c8a905f4270afc6de4c9e24fa337399c5656e0fabaf63a19e6c1860fd1.debug.log"  

    results = calculate_average_token_offset(input_log_file)

    if results is not None:
        print(f"Average Percent Difference (Actual - Predicted) / Actual: {results['average']:.2f}%")
        print(f"Minimum Percent Difference: {results['min']:.2f}%")
        print(f"    Predicted Tokens: {results['argmin_data']['predicted']}, Actual Tokens: {results['argmin_data']['actual']}, Line: {results['argmin_data']['line']}")
        print(f"Maximum Percent Difference: {results['max']:.2f}%")
        print(f"    Predicted Tokens: {results['argmax_data']['predicted']}, Actual Tokens: {results['argmax_data']['actual']}, Line: {results['argmax_data']['line']}")
        print(f"Standard Deviation of Percent Difference: {results['std_dev']:.2f}%")
