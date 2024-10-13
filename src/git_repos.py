
#git_repos.py


from llama_index.readers.github import GithubRepositoryReader

from llama_index.readers.github import GithubClient





>>> github_client = GithubClient(github_token=os.environ["GITHUB_TOKEN"], verbose=True)


client = github_client = GithubClient(
...    
...    verbose=True
... )
>>> reader = GithubRepositoryReader(
...    github_client=github_client,
...    owner="run-llama",
...    repo="llama_index",
... )
>>> branch_documents = reader.load_data(branch="branch")
>>> commit_documents = reader.load_data(commit_sha="commit_sha")
