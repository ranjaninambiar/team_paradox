export function dijkstra(grid, startNode, endNode) {
  const visitedNodesInOrder = [];
  startNode.distance = 0;
  const unvisitedNodes = getAllNodes(grid);
  while (!!unvisitedNodes.length) {
    sortNodesByDistance(unvisitedNodes);
    const shortestNode = unvisitedNodes.shift();
    if (shortestNode.isWall) continue;
    if (shortestNode.distance === Infinity) return visitedNodesInOrder;
    shortestNode.isVisited = true;
    visitedNodesInOrder.push(shortestNode);
    if (shortestNode === endNode) return visitedNodesInOrder;
    updateUnvisitedNeighbors(shortestNode, grid);
  }
}

function sortNodesByDistance(unvisitedNodes) {
  unvisitedNodes.sort((A, B) => A.distance - B.distance);
}

function updateUnvisitedNeighbors(node, grid) {
  const unvisitedNeighbors = [];
  const { col, row } = node;
  if (row > 0) {
    if (!grid[row - 1][col].isVisited)
      unvisitedNeighbors.push(grid[row - 1][col]);
  }
  if (row < grid.length - 1) {
    if (!grid[row + 1][col].isVisited)
      unvisitedNeighbors.push(grid[row + 1][col]);
  }
  if (col > 0) {
    if (!grid[row][col - 1].isVisited)
      unvisitedNeighbors.push(grid[row][col - 1]);
  }
  if (col < grid[0].length - 1) {
    if (!grid[row][col + 1].isVisited)
      unvisitedNeighbors.push(grid[row][col + 1]);
  }
  for (const neighbor of unvisitedNeighbors) {
    neighbor.distance = node.distance + 1;
    neighbor.previousNode = node;
  }
}
function getAllNodes(grid) {
  const nodes = [];
  for (const row of grid) {
    for (const node of row) {
      nodes.push(node);
    }
  }
  return nodes;
}
export function getShortestPathOrder(endNode) {
  const shortestPathOrder = [];
  let currNode = endNode;
  while (currNode !== null) {
    shortestPathOrder.unshift(currNode);
    currNode = currNode.previousNode;
  }
  return shortestPathOrder;
}
