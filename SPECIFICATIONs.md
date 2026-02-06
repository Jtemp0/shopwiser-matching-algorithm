# **Service Description: ShopWiser Algorithmic Product Categorisation**

**Provider:** Mihail Ionita

**Client:** Jack Templeton (ShopWiser Co-Founder)

## **1\. Executive Details**

* **Project Goal:** To programmatically restructure raw supermarket inventory data into "Low-Level Categories" that allow for direct price comparison across four target retailers: Tesco, ASDA, Morrisons, and Sainsbury’s.

* **Core Deliverable:** An algorithmic solution that ingests raw product data and outputs a structured set of \~10,000–20,000 normalized product clusters given the dataset of \~65,000 rows.

* **Target Domain:** Grocery items (specifically Food and Drink), filtered from the larger dataset.

## **2\. Problem Statement**

ShopWiser aims to allow users to compare the total price of a shopping basket across different supermarkets. Currently, this is impossible because retailers label identical products differently (e.g., *“Tesco Value Sweetcorn”* vs. *“Hubbards Sweetcorn”*) and raw data is categorized too broadly (e.g., just *“Drinks”*).

To function, the app requires a "universal layer" where user intent (e.g., "Chopped Tomatoes 400g") maps to the exact correct SKU at every supermarket.

**Key challenges include:**

* **Weight Sensitivity:** A search for "400g" must not return a cheaper "200g" alternative.

* **Semantic Nuance:** "Plum Tomatoes" must not be grouped with "Chopped Tomatoes" despite similar pricing and weight.

* **Brand Bias:** Store-specific branding (e.g., "Essential Waitrose") prevents simple text matching and must be normalized.

## 

## **3\. Intended Project Scope & Logic**

Mihail will develop an algorithm to process the provided raw dataset (65k relevant rows) and generate a relational map of products.

### **The Process:**

1. **Ingestion & Filtering:** Filter the raw dataset to isolate relevant "Food and Drink" categories, removing non-comparable household goods.

2. **Normalization (Data Cleaning):**  
   * Strip supermarket-specific prefixes (e.g., "Tesco", "Sainsbury's") to create "Clean Titles".  
   * Standardize weights and units (e.g., converting all to grams/liters).

3. **Algorithmic Clustering:** Group products into Low-Level Categories based on a strict matching of:  
   * **Core Identity:** (e.g., "Sweetcorn in Water").  
   * **Weight/Volume:** (e.g., "198g" or "200g").  
   * **Type/Attribute:** (e.g., "Chopped" vs. "Plum").

4. **Handling "Own Brand" vs. "Branded":**  
   * *Branded Goods:* Direct matching (e.g., "Heinz Beans 415g" is identical everywhere).  
   * *Own-Label Goods:* Creating clusters for "equivalent" tiers (e.g., grouping all "Value Range Spaghetti 500g" together).

## **4\. Success Criteria & Testing Logic**

Since a pre-defined "Truth Set" does not exist (creating it is the project), success will be measured via Statistical Sampling of the Output.

**Validation Protocol:**

* **Output Generation:** Mihail will deliver the full set of generated categories (estimated 10k–20k clusters).

* **Random Sampling:** The Client will randomly select 100 generated categories from the final output.

* **Manual Audit:** The Client will manually (or using LLM) verify these 100 clusters against the following "Pass" criteria:   
  * *Weight Accuracy:* Do all products in the cluster match the category weight? (e.g., No 200g tins mixed with 400g tins).  
  * *Type Accuracy:* Are the products semantically identical? (e.g., No "Plum" mixed with "Chopped").  
  * *Coverage:* Does the cluster contain a match for at least 3 of the 4 supermarkets (where applicable)?

**Threshold for Success:** The project is complete if more than 90% of the sampled clusters are free of "Critical Errors" (Wrong Weight, Wrong Product Type).