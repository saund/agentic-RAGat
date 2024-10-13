import os
import re
import ast

from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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

DEFAULT_RELEVANCY_PROMPT_TEMPLATE = """
    As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

    Retrieved Document:
    -------------------
    {context}

    User Question:
    --------------
    {query}

    Evaluation Criteria:
    - Consider whether the documents contains keywords or topics related to the user's question.
    - The evaluation should not be overly stringent; the primary objective is to find if all the context together is sufficient or not through scoring.

    Decision:
    - You *HAVE* to return a list of 2 strings. First string is the score. second string is the answer generated using the context
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

class TextExtractEvent(Event):
    """Text extract event. Extracts relevant text and concatenates."""

    relevant_text: str

class QueryEvent(Event):
    """Query event. Queries given relevant text and search text."""

    relevant_text: str
    search_text: str

class CorrectiveRAGWorkflow(Workflow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pinecone_key = os.getenv("PINECONE_API_KEY")
        index_name = "langchain-test-index-mohit"
        pc = Pinecone(api_key=pinecone_key)
        self.index = pc.Index(index_name)

    @step
    async def prepare_for_retrieval(
        self, ctx: Context, ev: StartEvent
    ) -> PrepEvent | None:
        """Prepare for retrieval."""

        
        query: str | None = ev.get("query")
        retriever_kwargs: dict | None = ev.get("retriever_kwargs", {})
        index: str | None = self.index

        llm = ChatOpenAI(model="gpt-4o")
        # print(f'Passed with index = {index}, query - {query}')

        print('c')
        prompt = PromptTemplate(template=DEFAULT_RELEVANCY_PROMPT_TEMPLATE, input_variables=["context", "query"])
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
        print('c')

        return PrepEvent()
    
    @step
    async def retrieve(
        self, ctx: Context, ev: PrepEvent
    ) -> RetrieveEvent | None:
        """Retrieve the relevant nodes for the query."""
        query = await ctx.get("query")

        print('ch')

        # retriever_kwargs = await ctx.get("retriever_kwargs")
        index = await ctx.get("index", default=None)
        if not index:
            raise ValueError(
                "Index must be constructed."
            )

        # retriever: BaseRetriever = index.as_retriever(**retriever_kwargs)

        retriever = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())

        result = retriever.similarity_search(query, k=5)

        await ctx.set("retrieved_result", result)
        await ctx.set("query", query)
        print('chhhhhhhh')
        return RetrieveEvent(retrieved_result=result)
    
    @step
    async def eval_relevance(
        self, ctx: Context, ev: RetrieveEvent
    ) -> StopEvent:
        """Evaluate relevancy of retrieved documents with the query."""

        print('chhc')
        retrieved_result = ev.retrieved_result
        query = await ctx.get("query")

        # relevancy_results = []
        print([doc for doc in retrieved_result])
        retrieved_results_text = [doc.page_content for doc in retrieved_result]
        relevancy_chain = await ctx.get("relevancy_chain")
        relevancy = relevancy_chain.invoke({
            "query": query,
            "context": retrieved_results_text
        }
        ).content

        match = re.search(r'\[(.*?)\]', relevancy)

        if match:
            # Extract the content inside the brackets
            list_string = match.group(0)  # This gets the whole match including brackets
            # Convert the string representation of the list to an actual list safely
            result_list = ast.literal_eval(list_string)
            print(result_list)  # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            print("No list found")

        await ctx.set("relevancy_results", relevancy)

        return StopEvent(result=relevancy)
    
async def main():
    c = CorrectiveRAGWorkflow(verbose=True)
    result = await c.run(query='hi')
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
