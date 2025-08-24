import re
import json

# Load the README file
with open("C:/Users/ibtis/Downloads/README.md", "r", encoding="utf-8") as f:
    readme_content = f.read()

# Regex patterns to capture categories and tools inside <details> sections
category_pattern = re.compile(r"<summary>(.*?)</summary>(.*?)</details>", re.S)
tool_pattern = re.compile(r"- \*\*(.*?)\*\* - (.*?)\n((?:\s{2,}- .*?\n)*)", re.S)
param_pattern = re.compile(r"- `([^`]+)`: (.*?)\n")

tools_data = {}

# Iterate through categories
for cat_match in category_pattern.finditer(readme_content):
    category = cat_match.group(1).strip()
    content = cat_match.group(2)

    tools = {}
    for tool_match in tool_pattern.finditer(content):
        tool_name = tool_match.group(1).strip()
        tool_desc = tool_match.group(2).strip()
        params_block = tool_match.group(3)

        params = {}
        for param_match in param_pattern.finditer(params_block):
            param_name = param_match.group(1).strip()
            param_desc = param_match.group(2).strip()
            params[param_name] = param_desc

        tools[tool_name] = {
            "description": tool_desc,
            "parameters": params
        }

    tools_data[category] = tools

# Save to JSON
output_path = "github_mcp_tools.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tools_data, f, indent=2)

output_path
