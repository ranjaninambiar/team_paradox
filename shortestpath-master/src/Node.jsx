import React from "react";
import "./nodeStyle.css";
export default class Node extends React.Component {
  constructor(props) {
    super(props);
    this.state = {};
  }

  render() {
    const {
      isEnd,
      isStart,
      row,
      col,
      isWall,
      onMouseDown,
      onDoubleClick
    } = this.props;
    const extraClassName = isEnd
      ? "node-end"
      : isStart
      ? "node-start"
      : isWall
      ? "node-wall"
      : "";

    return (
      <div
        id={`node-${row}-${col}`}
        className={`node ${extraClassName}`}
        onMouseDownCapture={() => onMouseDown(row, col)}
        onDoubleClick={() => onDoubleClick(row, col)}
      />
    );
  }
}
