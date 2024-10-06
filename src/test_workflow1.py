

#test_workflow1.py

import random


#>>> import asyncio
#>>> asyncio.run(main())

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)




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
    
