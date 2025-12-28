import pandas as pd

from core.ai_router import route_user_request
from core.cleaning_executor import execute_tool
from core.confirm import describe_tool_call, confirm_action, log_action
from core.checks import infer_all_column_types

print("✅ Script started")

df = pd.read_csv("sample_data/sales_bad.csv")
print("✅ CSV loaded")

column_types = infer_all_column_types(df)
print("✅ Column types resolved:", column_types)

user_input = "Fill missing values in ammount with mean"
print("👤 User input:", user_input)

tool_call = route_user_request(user_input, column_types)
print("🤖 AI tool call:", tool_call)

if tool_call:
    description = describe_tool_call(tool_call)
    print("📋 Proposed action:", description)

    if confirm_action(description):
        before = df["amount"].isna().sum()
        df = execute_tool(df, tool_call, column_types)
        after = df["amount"].isna().sum()

        log_action(tool_call)

        print(f"✅ Cleaning applied. Nulls before={before}, after={after}")
    else:
        print("🚫 Action cancelled")
else:
    print("❌ No tool suggested")
