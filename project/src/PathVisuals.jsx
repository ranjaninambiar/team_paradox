import React from "react";
import Node from "./Node";
import "./pathStyle.css";
import { dijkstra, getShortestPathOrder } from "./dijkstra";

const startRow = 2;
const startCol = 2;
const endRow = 9;
const endCol = 9;

export default class PathVisuals extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      nodes: [],
      mouseIsPressed: false
    };
    this.renderInitialGrid = this.renderInitialGrid.bind(this);
    this.buildWall = this.buildWall.bind(this);
  }
  componentDidMount() {
    this.renderInitialGrid();
  }

  renderInitialGrid() {
    const nodes = [];
    for (let row = 0; row < 22; row++) {
      const currow = [];
      for (let col = 0; col < 50; col++) {
        const node = {
          row,
          col,
          isStart: row === startRow && col === startCol,
          isEnd: row === endRow && col === endCol,
          distance: Infinity,
          isVisited: false,
          isWall: false,
          previousNode: null
        };
        currow.push(node);
      }
      nodes.push(currow);
    }
    this.setState({ nodes });
  }
  visualizeDijkstra() {
    const { nodes } = this.state;
    const startNode = nodes[startRow][startCol];
    const finishNode = nodes[endRow][endCol];
    const visitedNodesInOrder = dijkstra(nodes, startNode, finishNode);
    const shortestPathOrder = getShortestPathOrder(finishNode);
    this.animateDijkstra(visitedNodesInOrder, shortestPathOrder);
  }

  handleMouseDown(row, col) {
    const newGrid = getNewGridWithWallToggled(this.state.nodes, row, col);
    this.setState({ grid: newGrid, mouseIsPressed: true });
  }

  handleMouseEnter(row, col) {
    if (!this.state.mouseIsPressed) return;
    const newGrid = getNewGridWithWallToggled(this.state.nodes, row, col);
    this.setState({ grid: newGrid });
  }

  handleMouseUp() {
    this.setState({ mouseIsPressed: false });
  }
  animateDijkstra(visitedNodesInOrder, shortestPathOrder) {
    for (let i = 0; i <= visitedNodesInOrder.length; i++) {
      if (i === visitedNodesInOrder.length) {
        setTimeout(() => {
          this.animateShortestPath(shortestPathOrder);
        }, 200 * i);
        return;
      }
      setTimeout(() => {
        const node = visitedNodesInOrder[i];
        document.getElementById(`node-${node.row}-${node.col}`).className =
          "node node-visited";
      }, 200 * i);
    }
  }

  animateShortestPath(shortestPathOrder) {
    for (let i = 0; i < shortestPathOrder.length; i++) {
      setTimeout(() => {
        const node = shortestPathOrder[i];
        document.getElementById(`node-${node.row}-${node.col}`).className =
          "node node-shortest-path";
      }, 50 * i);
    }
  }
  buildWall(row, col) {
    const newGrid = this.state.nodes.slice();
    const node = newGrid[row][col];
    const newNode = {
      ...node,
      isWall: !node.isWall
    };
    newGrid[row][col] = newNode;
    this.setState({ nodes: newGrid });
  }
  render() {
    const { nodes } = this.state;
    return (
      <>
        <button onClick={() => this.visualizeDijkstra()}>
          Visualize Dijkstra's Algorithm
        </button>
        <div className="nodes">
          {nodes.map((row, rowIndex) => {
            return (
              <div key={rowIndex}>
                {row.map((node, nodeIndex) => {
                  const { isStart, isEnd, col, row, isWall, isVisited } = node;
                  return (
                    <Node
                      key={nodeIndex}
                      col={col}
                      row={row}
                      isStart={isStart}
                      isWall={isWall}
                      isVisited={isVisited}
                      isEnd={isEnd}
                    />
                  );
                })}
              </div>
            );
          })}
        </div>
      </>
    );
  }
}
const getNewGridWithWallToggled = (grid, row, col) => {
  const newGrid = grid.slice();
  const node = newGrid[row][col];
  const newNode = {
    ...node,
    isWall: !node.isWall
  };
  newGrid[row][col] = newNode;
  return newGrid;
};
