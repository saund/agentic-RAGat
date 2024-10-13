

#content_refinement_agent.py

import os
import sys
import asyncio
import time
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
#from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

gl_github_token = os.getenv('GITHUB_TOKEN')
gl_pinecone_api_token = os.getenv('PINECONE_API_TOKEN')

#imports for the context refinement agent below


############################################################
#
#Base RAG 
#



def load_docs_from_repo(repo_url, repo_owner):
    github_client = GithubClient(github_token=gl_github_token, verbose=False)

    reader = GithubRepositoryReader(
        github_client=github_client,
        owner="Infineon",
        repo="mtb-example-btsdk-low-power-20819",
        use_parser=False,
        verbose=True,
        # filter_directories=(
        #     ["docs"],
        #     GithubRepositoryReader.FilterType.INCLUDE,
        # ),
        filter_file_extensions=(
            [
                ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", "json", ".ipynb",
            ],
            GithubRepositoryReader.FilterType.EXCLUDE,
        ),
    )
    documents = reader.load_data(branch="master")
    return docuemnts



gl_embeddings = OpenAIEmbeddings()

gl_pinecone_index_name = "langchain-test-index-mohit"  # change if desired



def fetch_context_from_vdb_for_text(query_text, top_k = 10):
    pc = Pinecone(api_key=gl_pinecone_api_token)
    #existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    index = pc.Index(gl_pinecone_index_name)
    vector_store = PineconeVectorStore(index=index,
                                       embedding=gl_embeddings)
    retrieval_results = vector_store.similarity_search_with_relevance_scores(query_text, k=top_k)
    return retrieval_results


from langchain.prompts import PromptTemplate

gl_llm_prompt_text = """
Human: You are an expert at generating answers from the context given.

You are given a list of documents as context to use as reference while generating the answer.

Use the documents as reference. if you dont know say you dont know.

Relevant Documents:

<context>
{context}
</context>

<question>
{query}
</question>

Assistant:"""

gl_lc_doc_prompt_template = PromptTemplate(template=gl_llm_prompt_text,
                                           input_variables=["context", "query"])


gl_top_k = 10



def baseline_rag_on_query(query_text):
    global gl_top_k
    retrieval_results = fetch_context_from_vdb_for_text(query_text, top_k = gl_top_k)
    llm_result = call_llm_for_rag_answer(query_text, retrieval_results)
    result_content = llm_result.content
    return result_content
    
    
#returns a list of tuple: (llamaindex Document with fields,
#                               dict metadata:
#                                  {'file_name': str ...,
#                                   'file_path': str ...,
#                                   'url':       str ....,
#                                  }
#                               str page_content
def fetch_context_from_vdb_for_text(query_text, top_k = 10):
    pc = Pinecone(api_key=gl_pinecone_api_token)
    index = pc.Index(gl_pinecone_index_name)
    vector_store = PineconeVectorStore(index=index,
                                       embedding=gl_embeddings)

    retrieval_results = vector_store.similarity_search_with_relevance_scores(query_text, k=top_k)
    return retrieval_results


def call_llm_for_rag_answer(query, retrieval_results):
    #assemble list of retrieval context documents
    str_retrieval_context = format_retrieval_results_context(retrieval_results)
    
    llm = ChatOpenAI(
        model='gpt-4o',
        temperature=0.1,
        max_tokens=2048
    )
    chain = gl_lc_doc_prompt_template | llm
    lc_doc_prompt_dict = {'query': query,
                          'context': str_retrieval_context}
    llm_result = chain.invoke(lc_doc_prompt_dict)
    return llm_result
         


#get the prompt text, calling the PromptTemplatte object
def format_prompt(query, retrieval_results):
    str_retrieval_context = format_retrieval_results_context(retrieval_results)
    prompt_text = gl_lc_doc_prompt_template.format(query = query, context = str_retrieval_context)
    return prompt_text

#This takes a list of tuple (Document, score), and pulls out the page_content from each
#Document, and then assembles a single context str that can be passed as the
#context part of the LLM prompt.
def format_retrieval_results_context(retrieval_results):
    str_retrieval_context = ''
    number = 1
    for retrieval_result in retrieval_results:
        retrieval_text = retrieval_result[0].page_content
        str_retrieval_context += f"\n---\nContext Document {number}:\n"
        number += 1
        str_retrieval_context += f"{retrieval_text}\n"
    return str_retrieval_context




#
#
############################################################

############################################################
#
#refinement agent workflow
#


from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)


#top level call
def run_refinement_rag_on_query(query_text):
    global gl_top_k
    
    #list of tuple (Document, score)
    retrieval_results = fetch_context_from_vdb_for_text(query_text, top_k = gl_top_k)
    
    str_refined_context = agentic_refine_retrieval_context(query_text, retrieval_results)
    
    llm_result = call_llm_for_rag_answer_from_refined_context(query_text, str_refined_context)
    result_content = llm_result.content
    return result_content

#context_str is a single string already assembled for delivery to the LLM
def call_llm_for_rag_answer_from_refined_context(query, str_context):
    #assemble list of retrieval context documents
    
    llm = ChatOpenAI(
        model='gpt-4o',
        temperature=0.1,
        max_tokens=2048
    )
    chain = gl_lc_doc_prompt_template | llm
    lc_doc_prompt_dict = {'query': query,
                          'context': str_context}
    llm_result = chain.invoke(lc_doc_prompt_dict)
    return llm_result


def agentic_refine_retrieval_context(query_text, retrieval_results):

    try:
        # Get the current event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No running event loop
        print(f"creating new event loop")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    #return loop.run_until_complete(async_func())
    #asyncio.run( wf.run() )
    wf = LoopExampleWorkflow(verbose=False)
    str_context = loop.run_until_complete(wf.run())

    
     
    str_context = format_retrieval_results_context(retrieval_results)
    return str_context




class EvaluateEvent(Event):
    number: int
    count: int
    count = 0

class RequestGenerateEvent(Event):
    count: int


class LoopExampleWorkflow(Workflow):
    @step
    async def generate(self, ev:StartEvent | RequestGenerateEvent) -> EvaluateEvent:
        if hasattr(ev, 'count'):
            count = ev.count
        else:
            count = 0
        random_number = random.randint(0, 10)
        #print(f"  generate generated random_number: {random_number}")
        return EvaluateEvent(number=random_number, count=count)


    @step
    async def evaluate(self, ev:EvaluateEvent) -> RequestGenerateEvent | StopEvent:
        number = ev.number
        count = ev.count
        count += 1
        if number < 7:
            print(f"count: {count} number: {number} < 7 so trying again")
            return RequestGenerateEvent(count=count)
        else:
            return StopEvent(result=f"finally got number {number} >= 7 after {count} tries")
        

async def main():
    wf = LoopExampleWorkflow(verbose=False)
    result = await wf.run()
    print(result)
    






    

#
#
############################################################
    



################################################################################
#
#Archives
#

#wrong.  This is not what goes to the LLM
def format_prompt_wrong(query, retrieval_results):
    retrieval_context_list = []
    for retrieval_result in retrieval_results:
        retrieval_context = retrieval_result[0].page_content
        retrieval_context_list.append(retrieval_context)
    
    context = "\n".join(retrieval_context_list)  # Join the context list into a single string
    prompt = f"Query: {query}\nContext:\n{context}"
    return prompt
