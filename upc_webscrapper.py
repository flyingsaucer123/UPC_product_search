# Import necessary libraries
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, BaseTool
from dotenv import load_dotenv
import os
import json
import time
from datetime import datetime
import sys
import codecs
from markdown_scrape_tool import MarkdownScrapeTool  # Import the MarkdownScrapeTool

# Change the default encoding for standard output to utf-8
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Load environment variables
load_dotenv()

# Set up your OpenAI API key (for other tools)
openai_api_key = os.getenv("OPENAI_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"
os.environ["OPENAI_API_KEY"] = openai_api_key

# Set up the Serper API key
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")  # Serper API Key

# Define the Serper search tool and ScrapeWebsiteTool
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Instantiate the markdown scrape tool by wrapping the existing ScrapeWebsiteTool
markdown_scrape_tool = MarkdownScrapeTool(scrape_tool)

# Define a function to generate a dynamic output file name based on product details
def generate_output_filename(inputs):
    base_name = f"{inputs['manufacturer_name']}_{inputs['product_description']}"
    base_name = base_name.replace(" ", "_").replace("/", "_").replace('"', '')  # Remove double quotes
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"output/{base_name}_{timestamp}.txt"

# Define the UPC Search Agent
upc_search_agent = Agent(
    role="UPC Search Specialist",
    goal="Search for the UPC code based on the product description, manufacturer, and part number.",
    backstory="You are skilled at retrieving UPC codes based on product data from various online sources.",
    tools=[search_tool],
    allow_delegation=True,
    verbose=True,
    memory=True
)

# Define the Scraping Agent to use MarkdownScrapeTool
scraping_agent = Agent(
    role="Scraping Expert",
    goal="Scrape ONLY the relevant sections of web pages to extract UPC codes/SKU from search results.",
    backstory="You specialize in scraping only the relevant sections and extracting UPC Code/SKU from websites. You do not scrape unnecessary information.",
    tools=[markdown_scrape_tool],
    allow_delegation=False,
    verbose=True,
    memory=True
)

# Define the JSON Writer Agent
writer = Agent(
    role='JSON Writer',
    goal='Write JSON files based on input and final output based on the results of the scraping tasks.',
    backstory='Writes JSON files based on input and output (UPC code, SKU, product link, image link) into a JSON format.',
    verbose=True,
    memory=True
)

# Task for UPC Search with dynamic output file generation
def create_upc_search_task(inputs):
    output_file = generate_output_filename(inputs)
    return Task(
        description=(
            "Use the product details (manufacturer: {manufacturer_name}, description: {product_description}, part number: {manufacturer_part_number}) "
            "to search the web for a matching UPC code. Use all the product details for accurate search results."
        ),
        expected_output="List of relevant web links from search results.",
        tools=[search_tool],
        agent=upc_search_agent,
        output_file=output_file
    )

# Task for Scraping using MarkdownScrapeTool
upc_scraping_task = Task(
    description=(
        "Scrape only the relevant sections of web pages found during the search for UPC codes/SKU based on the search results. "
        "Make sure to extract the UPC or SKU code from the scraped data. A typical UPC barcode's 12 digits represent the product's unique identifier."
    ),
    expected_output="Scraped content formatted in markdown containing the UPC, SKU, product link, and image link.",
    tools=[markdown_scrape_tool],
    agent=scraping_agent
)

# Task to write input and final output to JSON (overwrite approach)
write_to_json_task = Task(
    description="Combine the input data (manufacturer: {manufacturer_name}, description: {product_description}, part number: {manufacturer_part_number}) and final scraping results (UPC, SKU, product link, image link) and write them to a JSON file (overwrite the file).",
    expected_output="A JSON file with the input data and the scraping results (overwrite the file).",
    agent=writer,
    output_file="output/output_upc_search.json"  # Save the file in the specified directory
)

# Define the crew (task sequence: search -> scrape -> write)
def create_crew(search_task):
    return Crew(
        agents=[upc_search_agent, scraping_agent, writer],
        tasks=[search_task, upc_scraping_task, write_to_json_task],
        process=Process.sequential,  # Running tasks in sequence: Search -> Scrape -> Write to File
        verbose=True,
        memory=True
)

# Function to load inputs from a JSON file
def load_inputs_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Delete the intermediate file at the start if it exists
intermediate_file_path = "output/output_upc_search.json"
if os.path.exists(intermediate_file_path):
    os.remove(intermediate_file_path)

# Load input data from a JSON file (containing multiple entries)
input_file = 'input/filtered_data1.json'  # Path to your input JSON file
inputs_list = load_inputs_from_json(input_file)

# String to collect all final results
final_results = []

# Loop through each input data point and run the crew for each one
for inputs in inputs_list:
    print(f"Processing: {inputs['manufacturer_name']} - {inputs['product_description']}")

    # Create the UPC Search task dynamically with a unique output file
    upc_search_task = create_upc_search_task(inputs)

    # Create the crew with the dynamic search task
    crew = create_crew(upc_search_task)

    # Run the crew with the current input
    result = crew.kickoff(inputs=inputs)

    # After each crew.kickoff, read the intermediate JSON file to collect the results
    if os.path.exists(intermediate_file_path) and os.path.getsize(intermediate_file_path) > 0:
        with open(intermediate_file_path, 'r') as temp_file:
            try:
                raw_data = temp_file.read()  # Read the file as a string
                final_results.append(raw_data)  # Append the raw JSON data as a string
            except Exception as e:
                print(f"Error reading the intermediate file: {e}")
    else:
        print(f"No valid data found in {intermediate_file_path}, skipping.")

    print(f"Finished processing: {inputs['manufacturer_name']} - {inputs['product_description']}")

# Join the final results as a single string (optional) or write as a final list
final_results_string = "[\n" + ",\n".join(final_results) + "\n]"

print("FINAL RESULTS printed below:::")
print(final_results_string)

# Write the final results to an output text file
with open("output/final_results.txt", 'w') as output_file:
    output_file.write(final_results_string)

# Optionally, delete the intermediate file after processing
if os.path.exists(intermediate_file_path):
    os.remove(intermediate_file_path)



