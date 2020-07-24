import React from "react";
import Node from "./Node";
import "./pathStyle.css";
import { dijkstra, getShortestPathOrder } from "./dijkstra";

export default class PathVisuals extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      nodes: [],
      startRow: null,
      startCol: null,
      endRow: null,
      endCol: null,
      generateNode: "start",
      startGen: false,
      endGen: false
    };
    this.renderInitialGrid = this.renderInitialGrid.bind(this);
    this.handleMouseDown = this.handleMouseDown.bind(this);
    this.handleNodeChoice = this.handleNodeChoice.bind(this);
  }
  componentDidMount() {
    this.renderInitialGrid();
  }

  renderInitialGrid() {
    const nodes = [];
    for (let row = 0; row < 17; row++) {
      const currow = [];
      for (let col = 0; col < 50; col++) {
        const node = {
          row,
          col,
          isStart: row === this.state.startRow && col === this.state.startCol,
          isEnd: row === this.state.endRow && col === this.state.endCol,
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
    const startNode = nodes[this.state.startRow][this.state.startCol];
    const finishNode = nodes[this.state.endRow][this.state.endCol];
    const visitedNodesInOrder = dijkstra(nodes, startNode, finishNode);
    const shortestPathOrder = getShortestPathOrder(finishNode);
    this.animateDijkstra(visitedNodesInOrder, shortestPathOrder);
  }

  animateDijkstra(visitedNodesInOrder, shortestPathOrder) {
    for (let i = 0; i <= visitedNodesInOrder.length; i++) {
      if (i === visitedNodesInOrder.length) {
        setTimeout(() => {
          this.animateShortestPath(shortestPathOrder);
        }, 75 * i);
        return;
      }
      setTimeout(() => {
        const node = visitedNodesInOrder[i];
        document.getElementById(`node-${node.row}-${node.col}`).className =
          "node node-visited";
      }, 75 * i);
    }
  }

  animateShortestPath(shortestPathOrder) {
    for (let i = 0; i < shortestPathOrder.length; i++) {
      setTimeout(() => {
        const node = shortestPathOrder[i];
        document.getElementById(`node-${node.row}-${node.col}`).className =
          "node node-shortest-path";
      }, 30 * i);
    }
  }
  handleMouseDown(row, col) {
    const newGrid = this.state.nodes.slice();
    const node = newGrid[row][col];
    const newNode = {
      ...node,
      isWall: !node.isWall
    };
    newGrid[row][col] = newNode;
    this.setState({ nodes: newGrid });
  }
  generateStartEnd(row, col) {
    if (this.state.generateNode === "start") {
      if (this.state.startGen === false) {
        const newGrid = this.state.nodes.slice();
        const node = newGrid[row][col];
        const newNode = {
          ...node,
          isStart: !node.isStart
        };
        newGrid[row][col] = newNode;
        this.setState({
          nodes: newGrid,
          startGen: true,
          startRow: row,
          startCol: col
        });
      } else {
        if (row === this.state.startRow && this.state.startCol === col) {
          const newGrid = this.state.nodes.slice();
          const node = newGrid[row][col];
          const newNode = {
            ...node,
            isStart: !node.isStart
          };
          newGrid[row][col] = newNode;
          this.setState({
            nodes: newGrid,
            startGen: false,
            startRow: null,
            startCol: null
          });
        }
      }
    } else {
      if (this.state.endGen === false) {
        const newGrid = this.state.nodes.slice();
        const node = newGrid[row][col];
        const newNode = {
          ...node,
          isEnd: !node.isEnd
        };
        newGrid[row][col] = newNode;
        this.setState({
          nodes: newGrid,
          endGen: true,
          endRow: row,
          endCol: col
        });
      } else {
        if (row === this.state.endRow && this.state.endCol === col) {
          const newGrid = this.state.nodes.slice();
          const node = newGrid[row][col];
          const newNode = {
            ...node,
            isEnd: !node.isEnd
          };
          newGrid[row][col] = newNode;
          this.setState({
            nodes: newGrid,
            endGen: false,
            endRow: null,
            endCol: null
          });
        }
      }
    }
  }
  handleNodeChoice(event) {
    this.setState({ generateNode: event.target.value });
  }

  render() {
    const { nodes } = this.state;
    return (
      <>
        <div className="panel">
          <h1>Dijkstras Visualise</h1>
          <div className="chooseNode">
            <label>
              <input
                type="radio"
                value="start"
                checked={this.state.generateNode === "start"}
                onChange={e => this.handleNodeChoice(e)}
              />
              Choose Start
            </label>
            <label>
              <input
                type="radio"
                value="end"
                checked={this.state.generateNode === "end"}
                onChange={e => this.handleNodeChoice(e)}
              />
              Choose End
            </label>
            <h6>Choose Start or End option and double click to create Start or End nodes. Click to create obstacles</h6>
          </div>
          <div>
            {this.state.endGen && this.state.startGen && (
              <button onClick={() => this.visualizeDijkstra()}>
                Visualise
              </button>
            )}
          </div>
        </div>
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
                      onMouseDown={(row, col) => this.handleMouseDown(row, col)}
                      onDoubleClick={(row, col) =>
                        this.generateStartEnd(row, col)
                      }
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
