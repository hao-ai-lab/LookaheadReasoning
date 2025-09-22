import os
import json
import re
import argparse
from typing import List, Dict, Any, Tuple

def extract_boxed_answer(text: str) -> str:
    """Extract the last boxed answer from text."""
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1] if matches else ""

def load_jsonl_files(directory: str) -> List[Dict[str, Any]]:
    """Load all jsonl files from a directory."""
    data = []
    file_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_count += 1
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
    return data, file_count

def calculate_metrics(data: List[Dict[str, Any]]) -> Tuple[float, float, int, int]:
    """Calculate accuracy and speed for given data."""
    # Calculate average speed
    speeds = [item.get('speed', 0) for item in data if 'speed' in item]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    
    # Calculate accuracy
    correct = 0
    total = 0
    total_tokens = 0
    total_thinks = 0
    
    for item in data:
        if 'gold' in item and 'full_text' in item:
            gold_answer = str(item['gold']).strip()
            predicted_answer = extract_boxed_answer(item['full_text']).strip()
            
            total_thinks += item['full_text'].count('</think>')
            total_tokens += len(item['generation_tokens'])
            # print(len(item['generation_tokens']) / item['time_taken'], item['speed'])
            if predicted_answer.isdigit() and gold_answer.isdigit():
                if int(gold_answer) == int(predicted_answer):
                    correct += 1
            total += 1
    avg_tokens = total_tokens / total if total > 0 else 0
    avg_thinks = total_thinks / total if total > 0 else 0
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, avg_speed, avg_tokens, avg_thinks, correct, total

def find_directories_with_prefix(prefix: str) -> List[str]:
    """Find all directories in current directory that start with the given prefix."""
    current_dir = os.getcwd()
    matching_dirs = []
    
    for item in os.listdir(current_dir):
        if os.path.isdir(item) and item.startswith(prefix):
            matching_dirs.append(item)
    
    return sorted(matching_dirs)

def analyze_results(prefix: str) -> None:
    """Analyze results from directories matching the prefix."""
    directories = find_directories_with_prefix(prefix)
    
    if not directories:
        print(f"No directories found with prefix '{prefix}'")
        return
    
    print(f"Found directories with prefix '{prefix}': {directories}")
    print()
    
    all_data = []
    
    print("Individual Directory Results:")
    print("-" * 50)
    
    for directory in directories:
        data, file_count = load_jsonl_files(directory)
        all_data.extend(data)
        
        accuracy, avg_speed, avg_tokens, avg_thinks, correct, total = calculate_metrics(data)
        
        print(f"Directory: {directory}")
        print(f"  Entries: {len(data)} (from {file_count} files)")
        print(f"  Accuracy: {correct}/{total} ({accuracy:.2f}%)")
        print(f"  Average tokens: {avg_tokens}")
        print(f"  Average thinks: {avg_thinks}")
        print(f"  Average speed: {avg_speed:.2f}")
        print()
    
    if not all_data:
        print("No data found")
        return
    
    # Calculate overall metrics
    overall_accuracy, overall_speed, avg_tokens, avg_thinks, overall_correct, overall_total = calculate_metrics(all_data)

    print("Overall Results:")
    print("-" * 20)
    print(f"Total entries: {len(all_data)}")
    print(f"Overall accuracy: {overall_correct}/{overall_total} ({overall_accuracy:.2f}%)")
    print(f"Overall average speed: {overall_speed:.2f}")
    print(f"Overall average tokens: {avg_tokens}")
    print(f"Overall average thinks: {avg_thinks}")

def main():
    parser = argparse.ArgumentParser(description='Analyze JSONL results from directories with matching prefix')
    parser.add_argument('prefix', help='Prefix to match directory names')
    
    args = parser.parse_args()
    analyze_results(args.prefix)

if __name__ == "__main__":
    main()
