import cyt_tutorial

graph = {
  '5' : ['3','7'],
  '3' : ['2', '4'],
  '7' : ['8'],
  '2' : [],
  '4' : ['8'],
  '8' : []
}

visited = set() # Set to keep track of visited nodes of graph.

# Driver Code
print("Following is the Depth-First Search")
cyt_tutorial.dfs(visited, graph, '5')