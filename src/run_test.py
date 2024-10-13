from llama_index.utils.workflow import draw_all_possible_flows
import asyncio
import actual_workflow as tw

# asyncio.run(tw.execute_loop("how to use Auxiliary flash. mean how to read or write api's..? is there a limitation on write cycle..? is this can be used for event log ..? is can we have file system like fat16,fat32, etc ..? can i get the sample code example.","langchain-test-index-mohit-2"))

draw_all_possible_flows(tw.CorrectiveRAGWorkflow, filename='test.html')