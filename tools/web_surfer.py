import os
import re
import uuid
import mimetypes
import pathlib
from transformers.agents import ReactJsonAgent, HfEngine
from transformers.agents.agents import Tool, DEFAULT_REACT_JSON_SYSTEM_PROMPT

import time
import requests
import pathvalidate
from urllib.parse import urljoin, urlparse, unquote
from typing import Any, Dict, List, Optional, Union, Tuple
from tools.mdconvert import MarkdownConverter, UnsupportedFormatException, FileConversionException
from serpapi import GoogleSearch
from pypdf import PdfReader
from markdownify import markdownify as md

from tools import FileInspectorTool


user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

browser_config = {
    "serpapi_key": os.environ["SERPAPI_API_KEY"],
    "viewport_size": 1024 * 5,
    "downloads_folder": "coding",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
    },
}

class SimpleTextBrowser:
    """(In preview) An extremely simple text-based web browser comparable to Lynx. Suitable for Agentic use."""

    def __init__(
            self,
            start_page: Optional[str] = None,
            viewport_size: Optional[int] = 1024 * 8,
            downloads_folder: Optional[Union[str, None]] = None,
            serpapi_key: Optional[Union[str, None]] = None,
            request_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    ):
        self.start_page: str = start_page if start_page else "about:blank"
        self.viewport_size = viewport_size  # Applies only to the standard uri types
        self.downloads_folder = downloads_folder
        self.history: List[Tuple[str, float]] = list()
        self.page_title: Optional[str] = None
        self.viewport_current_page = 0
        self.viewport_pages: List[Tuple[int, int]] = list()
        self.set_address(self.start_page)
        self.serpapi_key = serpapi_key
        self.request_kwargs = request_kwargs
        self._mdconvert = MarkdownConverter()
        self._page_content: str = ""

        self._find_on_page_query: Union[str, None] = None
        self._find_on_page_last_result: Union[int, None] = None  # Location of the last result

    @property
    def address(self) -> str:
        """Return the address of the current page."""
        return self.history[-1][0]

    def set_address(self, uri_or_path: str, filter_year: Optional[int] = None) -> None:
        # TODO: Handle anchors
        self.history.append((uri_or_path, time.time()))

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")
        elif uri_or_path.startswith("google:"):
            self._serpapi_search(uri_or_path[len("google:"):].strip(), filter_year=filter_year)
        else:
            if (
                    not uri_or_path.startswith("http:")
                    and not uri_or_path.startswith("https:")
                    and not uri_or_path.startswith("file:")
            ):
                if len(self.history) > 1:
                    prior_address = self.history[-2][0]
                    uri_or_path = urljoin(prior_address, uri_or_path)
                    # Update the address with the fully-qualified path
                    self.history[-1] = (uri_or_path, self.history[-1][1])
            self._fetch_page(uri_or_path)

        self.viewport_current_page = 0
        self.find_on_page_query = None
        self.find_on_page_viewport = None

    @property
    def viewport(self) -> str:
        """Return the content of the current viewport."""
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.page_content[bounds[0]: bounds[1]]

    @property
    def page_content(self) -> str:
        """Return the full contents of the current page."""
        return self._page_content

    def _set_page_content(self, content: str) -> None:
        """Sets the text content of the current page."""
        self._page_content = content
        self._split_pages()
        if self.viewport_current_page >= len(self.viewport_pages):
            self.viewport_current_page = len(self.viewport_pages) - 1

    def page_down(self) -> None:
        self.viewport_current_page = min(self.viewport_current_page + 1, len(self.viewport_pages) - 1)

    def page_up(self) -> None:
        self.viewport_current_page = max(self.viewport_current_page - 1, 0)

    def find_on_page(self, query: str) -> Union[str, None]:
        """Searches for the query from the current viewport forward, looping back to the start if necessary."""

        # Did we get here via a previous find_on_page search with the same query?
        # If so, map to find_next
        if query == self._find_on_page_query and self.viewport_current_page == self._find_on_page_last_result:
            return self.find_next()

        # Ok it's a new search start from the current viewport
        self._find_on_page_query = query
        viewport_match = self._find_next_viewport(query, self.viewport_current_page)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def find_next(self) -> None:
        """Scroll to the next viewport that matches the query"""

        if self._find_on_page_query is None:
            return None

        starting_viewport = self._find_on_page_last_result
        if starting_viewport is None:
            starting_viewport = 0
        else:
            starting_viewport += 1
            if starting_viewport >= len(self.viewport_pages):
                starting_viewport = 0

        viewport_match = self._find_next_viewport(self._find_on_page_query, starting_viewport)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def _find_next_viewport(self, query: str, starting_viewport: int) -> Union[int, None]:
        """Search for matches between the starting viewport looping when reaching the end."""

        if query is None:
            return None

        # Normalize the query, and convert to a regular expression
        nquery = re.sub(r"\*", "__STAR__", query)
        nquery = " " + (" ".join(re.split(r"\W+", nquery))).strip() + " "
        nquery = nquery.replace(" __STAR__ ", "__STAR__ ")  # Merge isolated stars with prior word
        nquery = nquery.replace("__STAR__", ".*").lower()

        if nquery.strip() == "":
            return None

        idxs = list()
        idxs.extend(range(starting_viewport, len(self.viewport_pages)))
        idxs.extend(range(0, starting_viewport))

        for i in idxs:
            bounds = self.viewport_pages[i]
            content = self.page_content[bounds[0]: bounds[1]]

            # TODO: Remove markdown links and images
            ncontent = " " + (" ".join(re.split(r"\W+", content))).strip().lower() + " "
            if re.search(nquery, ncontent):
                return i

        return None

    def visit_page(self, path_or_uri: str, filter_year: Optional[int] = None) -> str:
        """Update the address, visit the page, and return the content of the viewport."""
        self.set_address(path_or_uri, filter_year=filter_year)
        return self.viewport

    def _split_pages(self) -> None:
        # Do not split search results
        if self.address.startswith("google:"):
            self.viewport_pages = [(0, len(self._page_content))]
            return

        # Handle empty pages
        if len(self._page_content) == 0:
            self.viewport_pages = [(0, 0)]
            return

        # Break the viewport into pages
        self.viewport_pages = []
        start_idx = 0
        while start_idx < len(self._page_content):
            end_idx = min(start_idx + self.viewport_size, len(self._page_content))  # type: ignore[operator]
            # Adjust to end on a space
            while end_idx < len(self._page_content) and self._page_content[end_idx - 1] not in [" ", "\t", "\r", "\n"]:
                end_idx += 1
            self.viewport_pages.append((start_idx, end_idx))
            start_idx = end_idx

    def _serpapi_search(self, query: str, filter_year: Optional[int] = None) -> None:
        if self.serpapi_key is None:
            raise ValueError("Missing SerpAPI key.")

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
        }
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        search = GoogleSearch(params)
        results = search.get_dict()
        self.page_title = f"{query} - Search"

        if len(results['organic_results']) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            self._set_page_content(
                f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter.")
            return

        def _prev_visit(url):
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i][0] == url:
                    return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
            return ""

        web_snippets: List[str] = list()
        idx = 0
        if "organic_results" in results:
            for page in results["organic_results"]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{_prev_visit(page['link'])}{snippet}"

                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

        content = (
                f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
                + "\n\n".join(web_snippets)
        )

        self._set_page_content(content)

    def _fetch_page(self, url: str) -> None:
        download_path = ""
        try:
            if url.startswith("file://"):
                download_path = os.path.normcase(os.path.normpath(unquote(url[7:])))
                res = self._mdconvert.convert_local(download_path)
                self.page_title = res.title
                self._set_page_content(res.text_content)
            else:
                # Prepare the request parameters
                request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}
                request_kwargs["stream"] = True

                # Send a HTTP request to the URL
                response = requests.get(url, **request_kwargs)
                response.raise_for_status()

                # If the HTTP request was successful
                content_type = response.headers.get("content-type", "")

                # Text or HTML
                if "text/" in content_type.lower():
                    res = self._mdconvert.convert_response(response)
                    self.page_title = res.title
                    self._set_page_content(res.text_content)
                # A download
                else:
                    # Try producing a safe filename
                    fname = None
                    download_path = None
                    try:
                        fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                        suffix = 0
                        while os.path.exists(download_path) and suffix < 1000:
                            suffix += 1
                            base, ext = os.path.splitext(fname)
                            new_fname = f"{base}__{suffix}{ext}"
                            download_path = os.path.abspath(os.path.join(self.downloads_folder, new_fname))

                    except NameError:
                        pass

                    # No suitable name, so make one
                    if fname is None:
                        extension = mimetypes.guess_extension(content_type)
                        if extension is None:
                            extension = ".download"
                        fname = str(uuid.uuid4()) + extension
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                    # Open a file for writing
                    with open(download_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=512):
                            fh.write(chunk)

                    # Render it
                    local_uri = pathlib.Path(download_path).as_uri()
                    self.set_address(local_uri)

        except UnsupportedFormatException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileConversionException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileNotFoundError:
            self.page_title = "Error 404"
            self._set_page_content(f"## Error 404\n\nFile not found: {download_path}")
        except requests.exceptions.RequestException as request_exception:
            try:
                self.page_title = f"Error {response.status_code}"

                # If the error was rendered in HTML we might as well render it
                content_type = response.headers.get("content-type", "")
                if content_type is not None and "text/html" in content_type.lower():
                    res = self._mdconvert.convert(response)
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{res.text_content}")
                else:
                    text = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        text += chunk
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{text}")
            except NameError:
                self.page_title = f"Error"
                self._set_page_content(f"## Error\n\n{str(request_exception)}")


browser = SimpleTextBrowser(**browser_config)


def _browser_state() -> Tuple[str, str]:
    header = f"Address: {browser.address}\n"
    if browser.page_title is not None:
        header += f"Title: {browser.page_title}\n"

    current_page = browser.viewport_current_page
    total_pages = len(browser.viewport_pages)

    address = browser.address
    for i in range(len(browser.history)-2,-1,-1): # Start from the second last
        if browser.history[i][0] == address:
            header += f"You previously visited this page {round(time.time() - browser.history[i][1])} seconds ago.\n"
            break

    header += f"Viewport position: Showing page {current_page+1} of {total_pages}.\n"

    print("VIEWPORT LEN: ", len(browser.viewport))

    return (header, browser.viewport)


class SearchInformationTool(Tool):
    name = "informational_web_search"
    description = "Perform an INFORMATIONAL web search query then return the search results."
    inputs = {
        "query": {
            "type": "text",
            "description": "The informational web search query to perform."
        },
        "filter_year": {
            "type": "text",
            "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!"
        }
    }
    output_type = "text"

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        browser.visit_page(f"google: {query}", filter_year=filter_year)
        header, content = _browser_state()
        return header.strip() + "\n=======================\n" + content

#
# class NavigationalSearchTool(Tool):
#     name = "navigational_web_search"
#     description = "Perform a NAVIGATIONAL web search query then immediately navigate to the top result. Useful, for example, to navigate to a particular Wikipedia article or other known destination. Equivalent to Google's \"I'm Feeling Lucky\" button."
#     inputs = {"query": {"type": "text", "description": "The navigational web search query to perform."}}
#     output_type = "text"
#
#     def forward(self, query: str) -> str:
#         if USE_SERPAPI_BROWSER:
#             browser.visit_page(f"google: {query}")
#         else:
#             browser.visit_page(f"bing: {query}")
#
#         # Extract the first linl
#         m = re.search(r"\[.*?\]\((http.*?)\)", browser.page_content)
#         if m:
#             browser.visit_page(m.group(1))
#
#         # Return where we ended up
#         header, content = _browser_state()
#         return header.strip() + "\n=======================\n" + content


class VisitTool(Tool):
    name = "visit_page"
    description = "Visit a webpage at a given URL and return its text. This will not work if the page is a pdf or txt: in that case, use the download_file tool instead."
    inputs = {"url": {"type": "text", "description": "The url of the webpage to visit."}}
    output_type = "text"

    def forward(self, url: str) -> str:
        browser.visit_page(url)
        header, content = _browser_state()
        return header.strip() + "\n=======================\n" + content


def extract_text_from_pdf(pdf_path):
    pdf = PdfReader(pdf_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return md(text)


class DownloadTool(Tool):
    name = "download_file"
    description = "Download a file at a given URL and return its text. Use this to inspect a PDF or text file."
    inputs = {"url": {"type": "text", "description": "The url of the file to be downloaded."}}
    output_type = "text"

    def forward(self, url: str) -> str:
        if "arxiv" in url:
            url = url.replace("abs", "pdf")
        response = requests.get(url)
        if "pdf" in url:
            new_path = "/tmp/metadata.pdf"
        else:
            new_path = "/tmp/metadata.txt"

        with open(new_path, "wb") as f:
            f.write(response.content)

        if "pdf" in url:
            text = extract_text_from_pdf(new_path)
        else:
            text = ""
            with open(new_path, "r") as f:
                while True:
                    line = f.readline()
                    if (not line) or (len(text) > 70000):
                        break
                    text += line
        return text


class PageUpTool(Tool):
    name = "page_up"
    description = "Scroll the viewport UP one page-length in the current webpage and return the new viewport content."
    output_type = "text"

    def forward(self, ) -> str:
        browser.page_up()
        header, content = _browser_state()
        return header.strip() + "\n=======================\n" + content


class PageDownTool(Tool):
    name = "page_down"
    description = "Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content."
    output_type = "text"

    def forward(self, ) -> str:
        browser.page_down()
        header, content = _browser_state()
        return header.strip() + "\n=======================\n" + content


class WebSearchTool(Tool):
    name = "ask_search_agent"
    description = "A tool with access to a web browser. Use to perform web searches, open pages, navigate to specific pages (e.g. Wikipedia), answer questions from pages."

    inputs = {
        "query": {
            "description": "The web search query",
            "type": "text",
        }
    }
    output_type = "text"

    def __init__(
            self,
            llm_engine,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.agent = ReactJsonAgent(
            llm_engine=llm_engine,
            tools=[
                SearchInformationTool(),
                VisitTool(),
                DownloadTool(),
                PageUpTool(),
                PageDownTool(),
                FileInspectorTool(
                    llm_engine=llm_engine,
                    description="""Call this tool to read a downloaded file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""
                )
            ],
            max_iterations=12,
            verbose=1,
            system_prompt=DEFAULT_REACT_JSON_SYSTEM_PROMPT + "\nAdditionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.",
        )

    def forward(self, query: str) -> str:
        return self.agent.run(query)