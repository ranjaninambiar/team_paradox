import React from "react";
import "./nodeStyle.css";
export default class Node extends React.Component {
  constructor(props) {
    super(props);
    this.state = {};
  }

  render() {
    const { isEnd, isStart, row, col, isVisited, isWall } = this.props;
    const extraClassName = isEnd
      ? "node-end"
      : isStart
      ? "node-start"
      : isVisited
      ? "node-visited"
      : isWall
      ? "node-Wall"
      : "";

    return (
      <div id={`node-${row}-${col}`} className={`node ${extraClassName}`} />
    );
  }
}
