
import pandas as pd

file_path = "data/compphish/Mapping_File.xlsx"

# Show available Excel sheets
excel = pd.ExcelFile(file_path)

print("Available sheets:")
print(excel.sheet_names)

# Inspect each sheet
for sheet in excel.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)

    print(f"\nSHEET: {sheet}")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())
