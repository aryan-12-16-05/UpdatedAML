"""
Graph Visualization Module using Plotly & NetworkX
Renders interactive graph network topology for AML transaction analysis.
Supports 2-hop visualization with hover customer profiles.
"""

import plotly.graph_objects as go
import networkx as nx
import fraud_data


def _build_hover_text(node, data):
    """Build a rich hover tooltip for a given node."""
    node_type = data.get("node_type", "node")
    hop = data.get("hop", 0)

    def is_valid(v):
        return v is not None and str(v).strip() not in ["", "—", "None", "nan", "NaN"]

    lines = []
    if hop == 0:
        # Source sender node
        profile = fraud_data.get_customer_profile(node)
        lines.append("<b>🔴 SENDER ACCOUNT</b>")
        lines.append(f"<b>{profile.get('name', node)}</b>")
        lines.append(f"Account: {node}")
        
        for key, label in [('kyc_status', 'KYC'), ('risk_tier', 'Risk Tier'), ('city', 'City'), 
                           ('open_since', 'Open Since'), ('avg_tx_amount', 'Avg Tx Amt'), 
                           ('total_outgoing', 'Total Outgoing')]:
            val = profile.get(key)
            if is_valid(val):
                lines.append(f"{label}: {val}")

    elif hop == 1:
        # First-hop receiver
        profile = fraud_data.get_receiver_profile(node)
        risk_color = "🔴" if "High" in str(profile.get("risk_tier", "")) else "🟡"
        lines.append(f"<b>{risk_color} HOP-1 RECEIVER</b>")
        lines.append(f"<b>{profile.get('name', node)}</b>")
        lines.append(f"Account: {node}")
        
        for key, label in [('kyc_status', 'KYC'), ('risk_tier', 'Risk Tier'), ('city', 'City'), 
                           ('open_since', 'Open Since'), ('total_incoming', 'Total Incoming'), 
                           ('total_outgoing', 'Total Outgoing')]:
            val = profile.get(key)
            if is_valid(val):
                lines.append(f"{label}: {val}")
                
        notes = profile.get("notes")
        if is_valid(notes):
            lines.append(f"<i>{notes}</i>")

    elif hop == 2:
        # Second-hop node
        h2p = fraud_data.HOP2_PROFILES.get(node, {})
        lines.append("<b>⬡ HOP-2 NODE</b>")
        lines.append(f"<b>{h2p.get('name', node)}</b>")
        lines.append(f"Account: {node}")
        
        for key, label in [('kyc_status', 'KYC'), ('risk_tier', 'Risk Tier'), ('city', 'City')]:
            val = h2p.get(key)
            if is_valid(val):
                lines.append(f"{label}: {val}")
    else:
        lines.append(f"Account: {node}")
        if is_valid(node_type):
            lines.append(f"Role: {str(node_type).capitalize()}")

    return "<br>".join(lines)


def render_plotly_graph(tx_id, include_2hop=True):
    """
    Renders an interactive Plotly figure representing the transaction graph.
    Shows source → hop-1 receivers → hop-2 downstream nodes.
    Hovering on any node shows the customer profile.
    """
    G = fraud_data.create_network_graph(tx_id, include_2hop=include_2hop)
    pos = nx.spring_layout(G, seed=42, k=2.5)

    # ── Edge traces by hop ─────────────────────────────────────────────────
    edge_traces = []
    annotation_list = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        hop = edge[2].get("hop", 1)
        amount = edge[2].get("amount", "Transfer")

        color = "#ef4444" if hop == 1 else "#e2e8f0"
        width = 2.5 if hop == 1 else 0.8
        dash = "solid" if hop == 1 else "dot"

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color, dash=dash),
            hoverinfo="none",
            showlegend=False,
        ))

        # Mid-point label for hop-1 edges
        if hop == 1:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            annotation_list.append(dict(
                x=mx, y=my,
                text=f"<b>{amount}</b>",
                showarrow=False,
                font=dict(size=9, color="#dc2626"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="#fecaca",
                borderwidth=1,
                borderpad=2,
            ))

    # ── Node traces by type ────────────────────────────────────────────────
    node_groups = {"source": [], "hop1": [], "hop2": []}

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        hop = data.get("hop", 1)
        key = "source" if hop == 0 else ("hop2" if hop == 2 else "hop1")
        node_groups[key].append((node, data, x, y))

    node_traces = []

    group_styles = {
        "source": dict(
            name="Sender Account",
            symbol="star",
            size=38,
            color_fn=lambda d: d.get("color", "#ef4444"),
            line_color="#ffffff",
            line_width=3,
            text_color="#1e293b",
            opacity=0.95,
        ),
        "hop1": dict(
            name="Hop-1 Receivers",
            symbol="circle",
            size=28,
            color_fn=lambda d: d.get("color", "#f59e0b"),
            line_color="#ffffff",
            line_width=2,
            text_color="#334155",
            opacity=0.95,
        ),
        "hop2": dict(
            name="Hop-2 Nodes",
            symbol="diamond",
            size=15,
            color_fn=lambda d: "#f1f5f9",
            line_color="#cbd5e1",
            line_width=1,
            text_color="#94a3b8",
            opacity=0.65,
        ),
    }

    for key, nodes in node_groups.items():
        if not nodes:
            continue
        style = group_styles[key]
        xs = [n[2] for n in nodes]
        ys = [n[3] for n in nodes]
        labels = [n[0] for n in nodes]
        hover = [_build_hover_text(n[0], n[1]) for n in nodes]
        colors = [style["color_fn"](n[1]) for n in nodes]

        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            name=style["name"],
            text=labels,
            customdata=labels,
            textposition="top center",
            textfont=dict(size=9, color=style["text_color"]),
            hoverinfo="text",
            hovertext=hover,
            hoverlabel=dict(
                bgcolor="#1e293b",
                font_size=11,
                font_color="white",
                bordercolor="#475569",
            ),
            marker=dict(
                symbol=style["symbol"],
                size=style["size"],
                color=colors,
                line=dict(width=style["line_width"], color=style["line_color"]),
                opacity=style["opacity"],
            ),
            showlegend=True,
        ))

    all_traces = edge_traces + node_traces

    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            showlegend=True,
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                font=dict(size=11),
            ),
            hovermode="closest",
            margin=dict(b=30, l=20, r=20, t=50),
            annotations=annotation_list,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#f8fafc",
            plot_bgcolor="#f8fafc",
            height=520,
            title=dict(
                text=f"<b>Transaction Network — {tx_id}</b>   "
                     f"<span style='font-size:12px;color:#64748b;'>"
                     f"★ Sender  🟡 Hop-1 Receivers  ◆ Hop-2 Nodes  "
                     f"— Hover any node for customer profile</span>",
                font=dict(size=14, color="#1e293b"),
                x=0.0,
            ),
        )
    )

    return fig
