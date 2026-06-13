import ast
import re
import sys

def verify_files():
    files = ["code.md", "code_matcher.md", "code_matcher_2.md"]
    combined_code = []
    
    for filename in files:
        print(f"Reading {filename}...")
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            combined_code.append(content)
            
    full_code = "\n".join(combined_code)
    
    # Comment out Google Colab drive mount code to prevent syntax/import check failures on google.colab
    full_code = re.sub(r"from google\.colab import drive", "# from google.colab import drive", full_code)
    full_code = re.sub(r"drive\.mount\(.*?\)", "# drive.mount(...)", full_code)
    
    # Also comment out shell commands starting with ! (like !pip)
    full_code = re.sub(r"(?m)^(\s*)!(pip|zip|unzip|mkdir|rm|cp|mv|ls)", r"\1# \2", full_code)
    
    print("Parsing combined code with AST...")
    try:
        ast.parse(full_code)
        print("SUCCESS: Combined code is syntactically valid!")
        return True
    except SyntaxError as e:
        print(f"SYNTAX ERROR in concatenated code at line {e.lineno}, col {e.offset}:")
        print(f"Error message: {e.msg}")
        print(f"Line content: {e.text}")
        
        # Print a window around the error
        lines = full_code.splitlines()
        start = max(0, e.lineno - 10)
        end = min(len(lines), e.lineno + 10)
        for i in range(start, end):
            prefix = "--> " if i + 1 == e.lineno else "    "
            print(f"{prefix}{i+1}: {lines[i]}")
        return False

if __name__ == "__main__":
    success = verify_files()
    sys.exit(0 if success else 1)
