
#exploration.py

import os
import sys
import json





def getThreadsWithMention(thread_list, target_text = 'github'):
    ok_threads = []
    for i, thread in enumerate(thread_list):
        posts = thread.get('posts')
        if posts is not None:
            for post in posts:
                html = post.get('html')
                if html.find(target_text) >= 0:
                    print(f"{i}  {thread['source_id']}")
                    ok_threads.append(thread)
                    break
    return ok_threads



#no comments allowed, might also want to look at retriever_analysis_tools repo:  edm.load_ndjson
def readObjectFromLargeJsonFile(filepath):
    if not os.path.exists(filepath):
        print('cannot find file ' + filepath)
        return
    with open(filepath, 'r', encoding='utf-8') as fp:
        try:
            object = json.load(fp)
        except Exception as e:            
            print('Error: ' + str(e))
            return
    return object


#Writes a pseudo-ndjson file which is valid json:
#[
#{....},
#{....},
#{....}
#]
def writeObjectsToPseudoNDJsonFile(filepath, item_list):
    with open(filepath, 'w', encoding = 'utf-8') as file:
        file.write('[\n')
        for item in item_list[:-1]:
            str_item = json.dumps(item)
            file.write(str_item + ',\n')
        item = item_list[-1]
        str_item = json.dumps(item)
        file.write(str_item + '\n')
        file.write(']')
