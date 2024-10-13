import os
import re
import ast
import requests
import numpy as np
from uuid import uuid4

from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from collections import Counter

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)
# from llama_index.core import (
#     VectorStoreIndex,
#     SummaryIndex,
# )
from langchain_core.documents import Document

from langchain.prompts import PromptTemplate


from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.base.base_retriever import BaseRetriever

# from sklearn.metrics.pairwise import cosine_similarity

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from langchain_text_splitters import TokenTextSplitter

from llama_index.utils.workflow import draw_all_possible_flows


DEFAULT_RELEVANCY_PROMPT_TEMPLATE = """
    As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

    Retrieved Document:
    -------------------
    {context}

    User Question:
    --------------
    {query}

    Previous scored retreivals and answers:
    
    {previous_runs}

    Evaluation Criteria:
    - Consider whether the documents contains keywords or topics related to the user's question.
    - The evaluation should not be overly stringent; the primary objective is to find if all the context together is sufficient or not through scoring.

    Decision:

    - You *HAVE* to return a list of 2 strings. First string is the score. Second string is the answer generated using the context. 
    - previous runs of the same query is also given in the same List[str] format, if present. 
    - Use this as scoring criteria to see if answer is better / worse. 
    - Also use the answers to see if a better answer can be formed from context and previous answer
    - Make sure to give an elaborate answer when context is related and useful.

    Example output:
    1. ["0.5", "The documents provided does contain how to retrieve documents using Langchain but the code to do so is missing."]
    2. ["0.9", According to retrieved results.....]
    3. ["0.1", "Retrieved results does not answer the query asked"]
    """

class PrepEvent(Event):
    """Prep event (prepares for retrieval)."""
    pass
class RetrieveEvent(Event):
    """Retrieve event (gets retrieved results)."""

    retrieved_result: list[Document]

class RelevanceEvalEvent(Event):
    """Relevance evaluation event (gets results of relevance evaluation)."""

    relevant_results: list[str]

class RefineOnRepoEvent(Event):
    """Once decided on repo level refinement it comes here."""

class LinkParseEvent(Event):
    """Parse the Links, create index and search."""

    links: list

class QueryEvent(Event):
    """Query event. Queries given relevant text and search text."""

    relevant_text: str
    search_text: str

class RefinementEvent(Event):
    """ Refines Scratchpad to add or filter out relevant context"""
    pass

class RefineOnFolderEvent(Event):
    """ find folder and retrive full documents"""
    pass


class BaselineRAGWorkflow(Workflow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pinecone_key = os.getenv("PINECONE_API_KEY")
        index_name = "langchain-test-index-mohit-2"
        pc = Pinecone(api_key=pinecone_key)
        self.index = pc.Index(index_name)

    def best_5_links(self, links, query):

        embeddings = OpenAIEmbeddings()
        link_embeddings = embeddings.embed_documents(links)
        query_embedding = embeddings.embed_query(query)
        similarity_scores = cosine_similarity([query_embedding], link_embeddings)[0]
        top_5_indices = np.argsort(similarity_scores)[-5:][::-1]
        top_5_links = [links[i] for i in top_5_indices]

        return top_5_links

    def download_pdf(self, url, filename):

        response = requests.get(url, timeout=5)
        with open(f"{filename}.pdf", 'wb') as f:
            f.write(response.content)
        print(f"Downloaded PDF: {filename}.pdf")
        return

    def find_pdfs_and_download(self, links):

        files_downloaded=[]

        for url in links:

            try:
                response = requests.head(url, allow_redirects=True, timeout=5)  # Use head to get headers
                content_type = response.headers.get('Content-Type', '')

                if 'pdf' in content_type:
                    print(f"PDF found: {url}")
                    filename = url.split('/')[-1].split('.')[0]
                    files_downloaded.append(filename)
                    self.download_pdf(url, filename)  # Save with filename without extension
                else:
                    print(f"Other content type ({content_type}) for URL: {url}")
            except requests.RequestException as e:
                print(f"Error fetching {url}: {e}")
        return files_downloaded
    
    def parse_and_index_pdf_chunks(self, files_downloaded):

        parser = LlamaParse(
            result_type="text"  # "markdown" and "text" are available
        )

        file_extractor = {".pdf": parser}
        input_files=['./' + filename + '.pdf' for filename in a]
        pdf_documents = SimpleDirectoryReader(input_files=input_files, file_extractor=file_extractor).load_data()

        print(len(pdf_documents))

        updated_docs_pdf = []

        for document in pdf_documents:
            # Create a new Document instance with the updated attributes
            updated_document = Document(
                metadata=document.metadata,  # Keep metadata the same
                page_content=document.text,  # Change text to page_content
            )
            updated_docs_pdf.append(updated_document)

        chunks_pdf = []

        for doc in updated_docs_pdf:

            text_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=80)
            content = text_splitter.split_text(doc.page_content)
            for chunk in content:
                chunks_pdf.append(
                    Document(
                        metadata=doc.metadata,
                        page_content = chunk
                    ),
                    
                )

        uuids_pdf = [str(uuid4()) for doc in chunks_pdf]



    @step
    async def prepare_for_retrieval(
        self, ctx: Context, ev: StartEvent
    ) -> PrepEvent | None:
        """Prepare for retrieval."""

        query: str | None = ev.get("query")
        retriever_kwargs: dict | None = ev.get("retriever_kwargs", {})
        index: str | None = ev.get("index")
        count: int | None = ev.get("count",0)

        llm = ChatOpenAI(model="gpt-4o")
        # print(f'Passed with index = {index}, query - {query}')

        print('c')
        prompt = PromptTemplate(template=DEFAULT_RELEVANCY_PROMPT_TEMPLATE, input_variables=["context", "query", "previous_runs"])
        # prompt = 'hi'
        print('c')
        
        await ctx.set("llm", llm)
        await ctx.set("index", index)
        await ctx.set(
            "relevancy_chain", prompt | llm ,
        )
        print('c')
        await ctx.set("query", query)
        await ctx.set("retriever_kwargs", retriever_kwargs)
        await ctx.set("count", 0)
        print('c')

        return PrepEvent()
    
    @step
    async def retrieve(
        self, ctx: Context, ev: PrepEvent
    ) -> RetrieveEvent | None:
        """Retrieve the relevant chunks for the query."""
        query = await ctx.get("query")
        count = await ctx.get("count", default=0)
        chunk_stack = await ctx.get("chunk_stack",default=[])
        filter_for_rag = await ctx.get("filter_for_rag", default={})
        count+=1
        print(f'retrival count {count}')
        print(f'chunk_stack - {chunk_stack}')
        print(f'filter_for_rag - {filter_for_rag}')

        #index = await ctx.get("index", default=None)
        index = self.index
        print(f'index - {index}')
        if not index:
            raise ValueError(
                "Index must be constructed."
            )

        retriever = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())

        result = retriever.similarity_search(query, k=5, filter=filter_for_rag)

        await ctx.set("scratchpad", result)

        await ctx.set("retrieved_result", result)
        await ctx.set("query", query)
        # fake set count == 3 to skip to links retrieval
        await ctx.set("count", 0)
        chunk_stack.append(result)
        await ctx.set("chunk_stack", chunk_stack)

        stage = await ctx.get("stage",'')
        print(stage)
        # print(stage=='')
        if stage != '' and stage !='repo':
            print('came in to stop')
            return StopEvent()
        

        return RetrieveEvent(retrieved_result=result)
    
    @step
    async def eval_relevance(
        self, ctx: Context, ev: RetrieveEvent
    ) -> LinkParseEvent | PrepEvent | RefinementEvent | StopEvent:
        """Evaluate relevancy of retrieved documents with the query."""

        print('chhc')
        retrieved_result = ev.retrieved_result
        query = await ctx.get("query")
        previous_runs = await ctx.get("previous_runs", default=[])
        count = await ctx.get("count")
        print(f'retrieved results = {retrieved_result}')
        retrieved_results_text = [doc.page_content for doc in retrieved_result]
        relevancy_chain = await ctx.get("relevancy_chain")
        relevancy = relevancy_chain.invoke({
            "query": query,
            "context": retrieved_results_text,
            "previous_runs": previous_runs
        }
        ).content

        match = re.search(r'\[(.*?)\]', relevancy)

        if match:
            # Extract the content inside the brackets
            list_string = match.group(0)  # This gets the whole match including brackets
            # Convert the string representation of the list to an actual list safely
            result_list = ast.literal_eval(list_string)
            print(f'List found in response')
            print(f'relevance score - {result_list[0]}')
            print(f'Generated Answer - {result_list[1]}')
        else:
            print("No list found")

        await ctx.set("relevancy_results", result_list)
        
        # result_stack = await ctx.get("result_stack")
        # print(f'result_stack - {result_stack}')

        count = await ctx.get("count")

        if float(result_list[0]) >= 0.7:
            return StopEvent(result=result_list[1])
        else:
            return StopEvent(result=result_list[1])
        if count == 1:
            print(" Not Satisfied..Retrying RAG on a Repository level")
            # find the best repo to search
            repo_occurences = Counter([doc.metadata['repo'] for doc in retrieved_result])
            print(f'repo_occurences - {repo_occurences}')
            repo_count = repo_occurences.most_common()
            print(repo_count)
            await ctx.set("repo_count",repo_count) # would need if highest ocuring repo search gives significantly worse results
            print(f'repo count - {repo_count}')
            filter_for_rag = {"repo": repo_count[0][0]}
            await ctx.set("filter_for_rag", filter_for_rag)
            print(f'filter_for_rag is set as {filter_for_rag}')
            previous_runs.append(result_list)
            await ctx.set("previous_runs", previous_runs)
            print(f'previous runs  - {previous_runs}')
            return PrepEvent()
        elif count == 2:
            #### try using whole documents as context, regardless of repo search or folder search
            if float(previous_runs[-1][0]) - float(result_list[0]) > 0.2:

                # previous generation was better
                print('worse answer when first repo was generated, retrying')
                repo_count = await ctx.get("repo_count")
                if len(repo_count) == 1:
                    print('no more repos in initial retrieval, trying folder search...')
                    ## code for folder filter functionality - should include repo too
                    folder_to_search = Counter(doc.metadata['folder'] for doc in retrieved_result)
                    # repo info would already be present in retreived information this iteration
                    filter_for_rag = {"repo": retrieved_result[0].metadata['repo'], "folder": folder_to_search.most_common()[0][0]}
                    await ctx.set("filter_for_rag", filter_for_rag)
                    await ctx.set("index", "langchain-test-index-mohit-full-docs")

                else:
                    print('Trying another repo...')
                    filter_for_rag = {"repo": repo_count[1][0]}
                    await ctx.set("filter_for_rag", filter_for_rag)

                previous_runs.append(result_list)
                await ctx.set("previous_runs", previous_runs)
                return PrepEvent()
            else:
                # current generation is relatively similar / better
                folder_to_search = Counter(doc.metadata['folder'] for doc in retrieved_result)
                # repo info would already be present in retreived information this iteration
                filter_for_rag = {"repo": retrieved_result[0].metadata['repo'], "folder": folder_to_search.most_common()[0][0]}
                await ctx.set("filter_for_rag", filter_for_rag)
                await ctx.set("index", "langchain-test-index-mohit-full-docs")
                return PrepEvent()
        elif count==3:
            print('trying to see of there are links in the chunks')
            link_list = []
            chunk_stack = await ctx.get("chunk_stack")
            for chunk in chunk_stack:
                for doc in chunk:
                    links = re.findall(r'https?://[^\s)"]+(?=[)\s,."\'!])?', doc.page_content)
                    link_list += links
            top_5_links = self.best_5_links(link_list, query)
            print(f'Top 5 - {top_5_links}')
            return LinkParseEvent(links=top_5_links)
            return StopEvent(result=result_list[1])
            
        return StopEvent(result=result_list[1])
    
    @step
    async def refine_seratchpad(self, ctx: Context, ev: RefinementEvent) -> RefineOnRepoEvent | RefineOnFolderEvent | StopEvent:

        scratchpad = await ctx.get("scratchpad")
        stage = await ctx.get("stage",'')

        if stage == '':
            return RefineOnRepoEvent()
        elif stage == 'repo':
            return RefineOnFolderEvent()
        elif stage == 'folder':
            print('successfully came here')
        else:
            return StopEvent
        
    @step
    async def repo_refining(self, ctx: Context, ev: RefineOnRepoEvent) -> PrepEvent | StopEvent:

        scratchpad = await ctx.get("scratchpad")

        repo_occurences = Counter([doc.metadata['repo'] for doc in scratchpad])
        print(f'repo_occurences - {repo_occurences}')
        repo_count = repo_occurences.most_common()
        print(repo_count)
        await ctx.set("repo_count",repo_count) # would need if highest ocuring repo search gives significantly worse results
        print(f'repo count - {repo_count}')
        filter_for_rag = {"repo": repo_count[0][0]}
        await ctx.set("filter_for_rag", filter_for_rag)
        print(f'filter_for_rag is set as {filter_for_rag}')

        await ctx.set("stage", 'repo')

        return PrepEvent()

        return StopEvent(result='hi')
    
    @step
    async def folder_refining(self, ctx: Context, ev: RefineOnFolderEvent) -> StopEvent:
        print('came inside folder')
        scratchpad = await ctx.get("scratchpad")

        folders_or_files_to_pull = []

        for docs in scratchpad:
            folders_or_files_to_pull.append(docs.metadata['url'])

        

        raw_links = list(set([link.replace("https://github","https://raw.githubusercontent").replace("blob","refs/heads") for link in folders_or_files_to_pull]))
        print(f"raw links - {raw_links}")
        content = []
        for i in raw_links:

            response = requests.get(i)
            if response.status_code == 200:
                content.append(Document(page_content=response.text))
            else:
                print(f"Failed to retrieve {i}, status code: {response.status_code}")

        await ctx.set("stage", "folder")
        print(content)
        await ctx.set("scratchpad", content)
        # await ctx.set("",)

        return RetrieveEvent(retrieved_result=content)
    
    @step
    async def parse_links(
        self, ctx: Context, ev: LinkParseEvent
    ) -> StopEvent:
        
        links = ev.links
        files_downloaded = self.find_pdfs_and_download(links)
        # self.parse_and_index_pdf_chunks(files_downloaded)
        return StopEvent(result='hi')
    
async def main():
    query = """
    Group: Home/Forums/WirelessConnectivity/AIROC Bluetooth
Topic: cybt-213043-02 not entering sleep mode
Hi,
I am using cybt-213043-02 eval board. I am running https://github.com/Infineon/mtb-example-btsdk-low-power-20819 example with a tweak that changes the eval board to cybt-213043-eval.
I can see the output on the PUART when I press the user button and the ble works ok.
However the board never enters sleep mode. I see a constant ~2.5mA usage and the sleep permit handler is never called. Also sleep callback is never called.
The HOST_WAKE and DEV_WAKE pins are floating. I have tried tying them to either VCC or GND without effect. Is there some additional configuration or hw requirement to trigger sleep mode.
Thank you for response.
    """    
    c = BaselineRAGWorkflow(timeout=60.0, verbose=True)
    result = await c.run(query=query)
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
