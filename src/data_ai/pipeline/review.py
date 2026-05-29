# src/data_ai/pipeline/review.py
from pathlib import Path

from pyvis.network import Network

from data_ai.storage.models import Cluster
from data_ai.utils.similarity import cosine_similarity


def generate_review_html(
    clusters: list[Cluster],
    cluster_docs: dict[str, list[str]],
    output_path: Path,
    edge_threshold: float = 0.3,
) -> None:
    """
    Generate interactive HTML visualization of clusters.
    """
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
    )

    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=200)

    # Add nodes (clusters)
    for cluster in clusters:
        # Size based on doc count (min 20, max 100)
        size = min(100, max(20, cluster.doc_count * 5))

        # Color based on variance (green=low, red=high)
        variance_normalized = min(1.0, cluster.variance / 0.5)
        r = int(255 * variance_normalized)
        g = int(255 * (1 - variance_normalized))
        color = f"rgb({r},{g},100)"

        # Hover title with doc list
        docs = cluster_docs.get(cluster.id, [])
        docs_preview = docs[:10]
        if len(docs) > 10:
            docs_preview.append(f"... und {len(docs) - 10} weitere")

        title = f"""
<b>{cluster.name}</b><br>
Dokumente: {cluster.doc_count}<br>
Varianz: {cluster.variance:.2f}<br>
Status: {cluster.status.value}<br>
<hr>
{'<br>'.join(docs_preview)}
"""

        net.add_node(
            cluster.id,
            label=f"{cluster.name}\n({cluster.doc_count})",
            title=title,
            size=size,
            color=color,
        )

    # Add edges based on centroid similarity
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i+1:]:
            similarity = cosine_similarity(c1.centroid, c2.centroid)

            if similarity > edge_threshold:
                width = similarity * 5
                net.add_edge(
                    c1.id,
                    c2.id,
                    value=width,
                    title=f"Similarity: {similarity:.2f}",
                )

    # Generate HTML with custom additions
    net.save_graph(str(output_path))

    # Add summary table
    _add_summary_table(output_path, clusters, cluster_docs)


def _add_summary_table(
    html_path: Path,
    clusters: list[Cluster],
    cluster_docs: dict[str, list[str]],
) -> None:
    """Add a summary table to the HTML file."""
    content = html_path.read_text()

    table_html = """
<div style="margin: 20px; font-family: Arial, sans-serif;">
    <h2>Cluster-Übersicht</h2>
    <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #f0f0f0;">
            <th style="padding: 10px;">Name</th>
            <th style="padding: 10px;">Dokumente</th>
            <th style="padding: 10px;">Varianz</th>
            <th style="padding: 10px;">Status</th>
        </tr>
"""

    for cluster in sorted(clusters, key=lambda c: c.doc_count, reverse=True):
        status_color = {
            "proposed": "#ffd700",
            "approved": "#90ee90",
            "applied": "#add8e6",
        }.get(cluster.status.value, "#ffffff")

        table_html += f"""
        <tr>
            <td style="padding: 10px;"><b>{cluster.name}</b></td>
            <td style="padding: 10px; text-align: center;">{cluster.doc_count}</td>
            <td style="padding: 10px; text-align: center;">{cluster.variance:.2f}</td>
            <td style="padding: 10px; text-align: center; background-color: {status_color};">
                {cluster.status.value}
            </td>
        </tr>
"""

    table_html += """
    </table>
</div>
"""

    # Insert before closing body tag
    content = content.replace("</body>", f"{table_html}</body>")
    html_path.write_text(content)
