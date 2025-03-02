import os


def count_lines_in_directory(directory: str) -> int:
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return 0

    total_lines = 0

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines_in_file = sum(1 for _ in f)
                    total_lines += lines_in_file
            except Exception as e:
                print(f"Warning: Could not read '{file_path}': {e}")

    return total_lines


directory_path = "src"
total_lines = count_lines_in_directory(directory_path)
print(f'Total lines of code in "{directory_path}": {total_lines}')
