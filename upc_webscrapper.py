# Import necessary libraries
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, BaseTool
from dotenv import load_dotenv
import os
import json
import google.generativeai as genai  # Import Google Gemini API
import time
from datetime import datetime
import sys
import codecs

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

# Set up the Google Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define the Serper search tool and ScrapeWebsiteTool
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Custom Tool: Filter and clean scraped content via Google Gemini Pro 1.5 Flash with Retry
class CleanScrapedDataTool(BaseTool):
    name: str = "CleanScrapedDataTool"
    description: str = "This tool filters and cleans the scraped data by using a prompt-based approach via Gemini Pro 1.5 Flash and retries if necessary."

    def _run(self, scraped_data: str, retries: int = 1) -> str:
        attempt = 0
        cleaned_output = None

        while attempt <= retries:
            # Define the prompt for Gemini to clean the data
            prompt = f"""
            You have the following scraped content:
            {scraped_data}

            Please filter out unnecessary details and return only the following:
            - UPC code (if available)
            - SKU (if available)
            - Product link
            - Image link

            The output must be structured as valid JSON format.

            If a valid UPC is not found, try to extract other relevant identifiers (SKU, part number) and attempt one more time.
            """

            # Call Gemini Pro 1.5 Flash to process the prompt
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(response_mime_type="application/json")
            )

            # Parse the content returned by Gemini API
            if response.candidates and len(response.candidates) > 0:
                cleaned_output = response.candidates[0].content.parts[0].text

                # Check if UPC or SKU is found in the cleaned output
                if ('"UPC"' in cleaned_output or '"SKU"' in cleaned_output) and cleaned_output.count("null") == 0:
                    print(f"Attempt {attempt + 1}: Identifier found, stopping retries.")
                    break
                else:
                    print(f"Attempt {attempt + 1}: Identifier not found, retrying...")

            attempt += 1
            time.sleep(1)  # Small delay between retries

        return cleaned_output if cleaned_output else "Error: No valid response after retries from Gemini"

# Define a function to generate a dynamic output file name based on product details
def generate_output_filename(inputs):
    # Create a base name using the manufacturer name and product description
    base_name = f"{inputs['manufacturer_name']}_{inputs['product_description']}"
    
    # Replace any invalid characters in the filename (e.g., quotes, slashes, etc.)
    base_name = base_name.replace(" ", "_").replace("/", "_").replace('"', '')  # Remove double quotes
    
    # Add a timestamp to ensure unique file names
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Return the complete output file path
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

# Define the Scraping Agent
scraping_agent = Agent(
    role="Scraping Expert",
    goal="Scrape ONLY the relevant sections of web pages to extract UPC codes/SKU from search results.",
    backstory="You specialize in scraping only the relevant sections and extracting UPC Code/SKU from websites.You do not scrape unnecessary information",
    tools=[scrape_tool],
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

# Custom Agent: For processing scraped content using the custom tool with retry
cleaning_agent = Agent(
    role="Data Cleaner",
    goal="Clean and filter the scraped data using prompt-based methods and retry if needed.",
    backstory="You specialize in filtering and refining data using custom prompts and retrying when necessary.",
    tools=[CleanScrapedDataTool()],  # Use the custom tool here
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

# Task for Scraping
upc_scraping_task = Task(
    description=(
        "Scrape only the relevant sections of web pages found during the search for UPC codes/SKU based on the search results. "
        "Make sure to extract the UPC or SKU code from the scraped data. A typical UPC barcode's 12 digits represent the product's unique identifier."
    ),
    expected_output="Scraped content that contains the UPC, SKU, product link, and image link in a clean JSON format.",
    tools=[scrape_tool],  # Scraping the web pages returned from search
    agent=scraping_agent
)

# Task for Cleaning the Data using the prompt-based custom tool with retry
cleaning_task = Task(
    description="Use the scraped data and clean it by removing unnecessary details via Gemini Pro 1.5 Flash and retry up to 1 more time if no UPC or SKU code is not found.",
    expected_output="Cleaned JSON output with only necessary details (UPC, SKU, product link, image link).",
    tools=[CleanScrapedDataTool()],
    agent=cleaning_agent
)

# Task to write input and final output to JSON (overwrite approach)
write_to_json_task = Task(
    description="Combine the input data (manufacturer: {manufacturer_name}, description: {product_description}, part number: {manufacturer_part_number}) and final scraping results (UPC, SKU, product link, image link) and write them to a JSON file (overwrite the file).",
    expected_output="A JSON file with the input data and the scraping results (overwrite the file).",
    agent=writer,  # Writer agent handles combining and writing the result
    output_file="output/output_upc_search.json"  # Save the file in the specified directory
)

# Define the crew (task sequence: search -> scrape -> clean -> write)
def create_crew(search_task):
    return Crew(
        agents=[upc_search_agent, scraping_agent, cleaning_agent, writer],
        tasks=[search_task, upc_scraping_task, cleaning_task, write_to_json_task],
        process=Process.sequential,  # Running tasks in sequence: Search -> Scrape -> Clean -> Write to File
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
                # Read the contents of the intermediate file
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


