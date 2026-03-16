import pandas as pd

def convert_excel_to_formatted_csv(input_file, output_file):
    # Load the specific sheet
    # header=None is used initially to handle the category rows correctly
    df = pd.read_excel(input_file, sheet_name='Full Names and Links')

    # 1. Identify the header row (usually the one containing 'Name', 'Weight', etc.)
    # Based on the image, the real headers are in the first few rows.
    # We'll set the columns manually based on your spreadsheet structure.
    column_names = [
        "Product_Name", "Weight", 
        "Tesco_Name", "Tesco_Link", 
        "Sainsburys_Name", "Sainsburys_Link", 
        "Asda_Name", "Asda_Link", 
        "Morrisons_Name", "Morrisons_Link", 
        "Ocado_Name", "Ocado_Link"
    ]
    
    # Trim the dataframe if it has extra columns and assign names
    df = df.iloc[:, :len(column_names)]
    df.columns = column_names

    # 2. Logic to extract Categories
    # Categories are rows where 'Product_Name' is filled, but 'Weight' or 'Links' are NaN
    current_category = "Unknown"
    formatted_rows = []

    for index, row in df.iterrows():
        name = str(row['Product_Name']).strip()
        
        # Skip empty rows
        if not name or name == 'nan':
            continue
            
        # If 'Weight' is empty, this is a Category Header
        if pd.isna(row['Weight']) or str(row['Weight']).strip() == 'nan':
            current_category = name
            continue # Don't add the header row itself as a product
        
        # Add the category to the row data
        new_row = row.to_dict()
        new_row['Category'] = current_category
        formatted_rows.append(new_row)

    # 3. Create the final DataFrame
    final_df = pd.DataFrame(formatted_rows)

    # Reorder columns so Category is first (cleaner for CSVs)
    cols = ['Category'] + [c for c in column_names]
    final_df = final_df[cols]

    # 4. Save to CSV
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Success! Formatted CSV saved as: {output_file}")

if __name__ == "__main__":
    # Change 'grocery_list.xlsx' to your actual filename
    convert_excel_to_formatted_csv('grocery_list.xlsx', 'formatted_grocery_database.csv')