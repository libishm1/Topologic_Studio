import React from "react";

export default function Sidebar({ selectedFace }) {
  if (!selectedFace) {
    return (
      <aside className="sidebar">
        <h2>Inspector</h2>
        <p>Select a face to view its properties.</p>
      </aside>
    );
  }

  const { id, vertex_ids, data } = selectedFace;

  return (
    <aside className="sidebar">
      <h2>Inspector</h2>
      <div className="inspector-section">
        <h3>Face</h3>
        <p>
          <strong>ID:</strong> {id}
        </p>
        <p>
          <strong>Vertex IDs:</strong> {vertex_ids.join(", ")}</p>
      </div>

      <div className="inspector-section">
        <h3>Dictionary (data)</h3>
        {data ? (
          <pre>{JSON.stringify(data, null, 2)}</pre>
        ) : (
          <p>No data attached.</p>
        )}
      </div>
    </aside>
  );
}
