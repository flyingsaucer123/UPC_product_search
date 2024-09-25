from crewai_tools import BaseTool, ScrapeWebsiteTool

class MarkdownScrapeTool(BaseTool):
    name: str = "MarkdownScrapeTool"
    description: str = "A tool to scrape a webpage and return results formatted as markdown."

    def __init__(self, scrape_tool: ScrapeWebsiteTool):
        # Initialize the BaseTool
        super().__init__()
        # Store the scrape tool as a normal instance variable (not Pydantic-validated)
        self._scrape_tool = scrape_tool

    def _run(self, url: str) -> str:
        # Call the ScrapeWebsiteTool to scrape the webpage
        result = self._scrape_tool._run(url)
        
        # Format the result into markdown
        markdown_result = self._format_as_markdown(result)
        
        return markdown_result

    def _format_as_markdown(self, text: str) -> str:
        # Convert scraped result to markdown format
       # markdown_text = f"# Scraped Content from {text[:1000]}...\n\n"
      #  markdown_text += f"**Summary:**\n{text[:500]}...\n\n"  # Limit to first 500 characters
      #  markdown_text += "*For more details, check the original content.*"
       # return markdown_text
       return text
