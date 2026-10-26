"""
Graph Visualization Module using Plotly & NetworkX
Renders interactive graph network topology for AML transaction analysis.
Supports full 2-hop visualization with distinct color coding per hop:
- Hop 0 (Sender): Crimson Red Star
- Hop 1 (Direct Receivers): Vibrant Amber Circle
- Hop 2 (Downstream Layering): Slate Grey Diamond
"""

import math
import plotly.graph_objects as go
import networkx as nx
import fraud_data


def _build_hover_text(node, data):
    """Build a rich, informative hover tooltip for a given node."""
    hop = data.get("hop", 0)

    def is_valid(v):
        return v is not None and str(v).strip() not in ["", "—", "None", "nan", "NaN"]

    lines = []
    if hop == 0:
        profile = fraud_data.get_customer_profile(node)
        entity_name = data.get("entity_name") or profile.get("name", node)
        lines.append("<b>🔴 SENDER ACCOUNT (Hop 0)</b>")
        lines.append(f"<b>{entity_name}</b>")
        lines.append(f"Account: <code>{node}</code>")
        lines.append(f"Bank: {data.get('bank_name', 'Global Bank')}")
        if is_valid(profile.get("kyc_status")):
            lines.append(f"KYC Status: {profile.get('kyc_status')}")
        if is_valid(profile.get("risk_tier")):
            lines.append(f"Risk Tier: {profile.get('risk_tier')}")
        if is_valid(data.get("gat_prob")):
            lines.append(f"GAT Risk Prob: {float(data.get('gat_prob')):.1%}")
        if is_valid(profile.get("total_outgoing")):
            lines.append(f"Total Outgoing: {profile.get('total_outgoing')}")

    elif hop == 1:
        profile = fraud_data.get_receiver_profile(node)
        entity_name = data.get("entity_name") or profile.get("name", node)
        lines.append("<b>🟡 HOP-1 DIRECT RECEIVER</b>")
        lines.append(f"<b>{entity_name}</b>")
        lines.append(f"Account: <code>{node}</code>")
        lines.append(f"Bank: {data.get('bank_name', 'Global Bank')}")
        if is_valid(data.get("amount")):
            lines.append(f"Received from Sender: <b>{data.get('amount')}</b>")
        if is_valid(profile.get("kyc_status")):
            lines.append(f"KYC Status: {profile.get('kyc_status')}")
        if is_valid(profile.get("risk_tier")):
            lines.append(f"Risk Tier: {profile.get('risk_tier')}")
        if is_valid(profile.get("city")):
            lines.append(f"City: {profile.get('city')}")

    elif hop == 2:
        entity_name = data.get("entity_name", node)
        lines.append("<b>🔘 HOP-2 DOWNSTREAM LAYERING</b>")
        lines.append(f"<b>{entity_name}</b>")
        lines.append(f"Account: <code>{node}</code>")
        lines.append(f"Bank: {data.get('bank_name', 'International Clearing Bank')}")
        if is_valid(data.get("layering_type")):
            lines.append(f"Layering Type: <b>{data.get('layering_type')}</b>")
        if is_valid(data.get("amount")):
            lines.append(f"Downstream Inflow: <b>{data.get('amount')}</b>")
        if is_valid(data.get("parent_receiver")):
            lines.append(f"Forwarded from (Hop-1): <code>{data.get('parent_receiver')}</code>")
    else:
        lines.append(f"Account: <code>{node}</code>")

    return "<br>".join(lines)


def render_plotly_graph(tx_id, include_2hop=True):
    """
    Renders an interactive Plotly figure representing the transaction graph.
    Shows source (Hop 0) → direct receivers (Hop 1) → downstream layering (Hop 2).
    Each hop is visually distinguished with distinct colors, symbols, and edge styles.
    """
    G = fraud_data.create_network_graph(tx_id, include_2hop=include_2hop)
    
    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(title="No graph data found for this transaction group")
        return fig

    # ── Hierarchical Radial Layout (Hop 0 Center → Hop 1 Ring 1 → Hop 2 Ring 2) ──
    source_nodes = [n for n, d in G.nodes(data=True) if d.get("hop", 0) == 0]
    hop1_nodes = [n for n, d in G.nodes(data=True) if d.get("hop", 0) == 1]
    hop2_nodes = [n for n, d in G.nodes(data=True) if d.get("hop", 0) == 2]

    pos = {}
    if source_nodes:
        pos[source_nodes[0]] = (0.0, 0.0)

    n1 = len(hop1_nodes)
    for i, n in enumerate(hop1_nodes):
        angle = 2.0 * math.pi * i / max(n1, 1)
        r1 = 1.0
        pos[n] = (r1 * math.cos(angle), r1 * math.sin(angle))

    # Map each hop-2 node to its parent (Hop-1)
    hop2_by_parent = {}
    for u, v, edata in G.edges(data=True):
        if edata.get("hop") == 2:
            hop2_by_parent.setdefault(u, []).append(v)

    r2 = 2.25
    for parent, children in hop2_by_parent.items():
        if parent in pos:
            px, py = pos[parent]
            base_angle = math.atan2(py, px)
            k = len(children)
            for j, child in enumerate(children):
                if k == 1:
                    child_angle = base_angle
                else:
                    spread = 0.32
                    child_angle = base_angle - (spread / 2) + (j * spread / (k - 1))
                pos[child] = (r2 * math.cos(child_angle), r2 * math.sin(child_angle))

    # Fallback for any unpositioned nodes
    if len(pos) < len(G.nodes()):
        spring_pos = nx.spring_layout(G, seed=42, k=2.0)
        for n in G.nodes():
            if n not in pos:
                pos[n] = spring_pos.get(n, (0.0, 0.0))

    # ── Edge traces by hop ─────────────────────────────────────────────────
    edge_traces = []
    annotation_list = []

    for edge in G.edges(data=True):
        u, v = edge[0], edge[1]
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        hop = edge[2].get("hop", 1)
        amount = edge[2].get("amount", "Transfer")

        if hop == 1:
            color = "#ef4444"
            width = 2.5
            dash = "solid"
            badge_font = "#dc2626"
            badge_border = "#fecaca"
        else:
            color = "#94a3b8"
            width = 1.8
            dash = "dash"
            badge_font = "#475569"
            badge_border = "#cbd5e1"

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color, dash=dash),
            hoverinfo="none",
            showlegend=False,
        ))

        # Mid-point label for transfer amount
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        annotation_list.append(dict(
            x=mx, y=my,
            text=f"<b>{amount}</b>",
            showarrow=False,
            font=dict(size=8, color=badge_font),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=badge_border,
            borderwidth=1,
            borderpad=2,
        ))

    # ── Node traces by type ────────────────────────────────────────────────
    node_groups = {"source": [], "hop1": [], "hop2": []}

    for node, data in G.nodes(data=True):
        if node not in pos:
            continue
        x, y = pos[node]
        hop = data.get("hop", 1)
        key = "source" if hop == 0 else ("hop2" if hop == 2 else "hop1")
        node_groups[key].append((node, data, x, y))

    node_traces = []

    group_styles = {
        "source": dict(
            name="★ Sender Account (Hop 0)",
            symbol="star",
            size=36,
            color="#ef4444",
            line_color="#ffffff",
            line_width=3,
            text_color="#991b1b",
            opacity=1.0,
        ),
        "hop1": dict(
            name="● Hop-1 Direct Receivers",
            symbol="circle",
            size=26,
            color="#f59e0b",
            line_color="#ffffff",
            line_width=2.5,
            text_color="#92400e",
            opacity=0.95,
        ),
        "hop2": dict(
            name="◆ Hop-2 Downstream Layering",
            symbol="diamond",
            size=20,
            color="#64748b",
            line_color="#ffffff",
            line_width=2,
            text_color="#334155",
            opacity=0.95,
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
                bgcolor="#0f172a",
                font_size=11,
                font_color="white",
                bordercolor="#334155",
            ),
            marker=dict(
                symbol=style["symbol"],
                size=style["size"],
                color=style["color"],
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
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#cbd5e1",
                borderwidth=1,
                font=dict(size=11),
            ),
            hovermode="closest",
            margin=dict(b=30, l=20, r=20, t=55),
            annotations=annotation_list,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#f8fafc",
            plot_bgcolor="#f8fafc",
            height=580,
            title=dict(
                text=f"<b>Transaction Network Topology — {tx_id}</b>   "
                     f"<span style='font-size:12px;color:#64748b;'>"
                     f"🔴 Hop-0 Sender · 🟡 Hop-1 Receivers · 🔘 Hop-2 Downstream Layering</span>",
                font=dict(size=14, color="#0f172a"),
                x=0.0,
            ),
        )
    )

    return fig
