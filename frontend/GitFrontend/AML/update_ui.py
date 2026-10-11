import sys

file_path = 'C:/Users/aryan/frontend/GitFrontend/AML/app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
for i, line in enumerate(lines):
    if 'elif page == "Metrics":' in line:
        start_idx = i
        break

if start_idx != -1:
    new_content = lines[:start_idx]
    
    metrics_ui = """elif page == "Metrics":
    st.html(textwrap.dedent('''
    <div class="center-card" style="margin-bottom:20px;">
        <h2 style="margin:0;color:#0f172a;font-weight:800;">GAT Model Architecture & Metrics</h2>
        <p style="color:#64748b;font-size:15px;margin-top:5px;">
            Detailed layer-by-layer breakdown of the Graph Attention Network and performance metrics across training and testing data.
        </p>
    </div>
    '''))

    tab_arch, tab_train, tab_test = st.tabs(["🧠 Model Architecture", "📊 Training Metrics (7M)", "📊 Testing Metrics (1.5M)"])

    with tab_arch:
        st.markdown("### 🧬 GATv2 AML Model Architecture")
        st.markdown("The Edge-Level AML classifier utilizes node features, graph connectivity, and edge features to output a laundering risk probability for target transactions.")
        
        st.html(textwrap.dedent('''
        <div style="background:white;padding:20px;border-radius:12px;border:1px solid #e2e8f0;margin-bottom:20px;border-left:4px solid #3b82f6;box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <h4 style="margin-top:0;color:#1e40af;font-size:16px;">1. Node Feature Projection (Encoder)</h4>
            <div style="color:#475569;font-size:14px;margin-bottom:6px;"><strong>Layer:</strong> <code>nn.Linear(node_in_dim, 64)</code> + <code>LeakyReLU(negative_slope=0.2)</code></div>
            <div style="color:#64748b;font-size:13px;">Projects the raw 13 node features (like transaction counts, amounts) into a 64-dimensional hidden representation.</div>
        </div>

        <div style="background:white;padding:20px;border-radius:12px;border:1px solid #e2e8f0;margin-bottom:20px;border-left:4px solid #8b5cf6;box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <h4 style="margin-top:0;color:#5b21b6;font-size:16px;">2. Graph Attention Layers (GATv2Conv)</h4>
            
            <div style="margin-bottom:15px;">
                <div style="color:#475569;font-size:14px;font-weight:600;margin-bottom:4px;">GAT Layer 1</div>
                <div style="color:#475569;font-size:13px;padding-left:15px;border-left:2px solid #cbd5e1;margin-bottom:4px;">
                    <code>GATv2Conv(in_channels=64, out_channels=16, heads=4, concat=True, edge_dim=20)</code>
                </div>
                <div style="color:#64748b;font-size:13px;padding-left:15px;border-left:2px solid #cbd5e1;"><strong>Regularization:</strong> Residual Connection (h + h1) &rarr; <code>LayerNorm(64)</code> &rarr; <code>Dropout(0.2)</code></div>
            </div>

            <div>
                <div style="color:#475569;font-size:14px;font-weight:600;margin-bottom:4px;">GAT Layer 2</div>
                <div style="color:#475569;font-size:13px;padding-left:15px;border-left:2px solid #cbd5e1;margin-bottom:4px;">
                    <code>GATv2Conv(in_channels=64, out_channels=16, heads=4, concat=True, edge_dim=20)</code>
                </div>
                <div style="color:#64748b;font-size:13px;padding-left:15px;border-left:2px solid #cbd5e1;"><strong>Regularization:</strong> Residual Connection (h + h2) &rarr; <code>LayerNorm(64)</code></div>
            </div>
        </div>

        <div style="background:white;padding:20px;border-radius:12px;border:1px solid #e2e8f0;margin-bottom:20px;border-left:4px solid #ef4444;box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <h4 style="margin-top:0;color:#991b1b;font-size:16px;">3. Edge Classifier (Decoder)</h4>
            <div style="color:#475569;font-size:14px;margin-bottom:10px;">
                <strong>Input Concatenation:</strong> Source Node (64) + Destination Node (64) + Target Edge Features (20) = <strong>148 Dimensions</strong>
            </div>
            
            <div style="color:#475569;font-size:13px;padding-left:15px;border-left:2px solid #cbd5e1;margin-bottom:8px;">
                <strong>Dense Block 1:</strong> <code>nn.Linear(148, 64)</code> &rarr; <code>LayerNorm(64)</code> &rarr; <code>LeakyReLU(0.2)</code> &rarr; <code>Dropout(0.2)</code>
            </div>
            <div style="color:#475569;font-size:13px;padding-left:15px;border-left:2px solid #cbd5e1;margin-bottom:8px;">
                <strong>Dense Block 2:</strong> <code>nn.Linear(64, 32)</code> &rarr; <code>LeakyReLU(0.2)</code> &rarr; <code>Dropout(0.2)</code>
            </div>
            <div style="color:#475569;font-size:13px;padding-left:15px;border-left:2px solid #cbd5e1;margin-bottom:12px;">
                <strong>Output Logits:</strong> <code>nn.Linear(32, 1)</code>
            </div>

            <div style="background:#fef2f2;padding:10px;border-radius:6px;border:1px solid #fecaca;color:#991b1b;font-size:13px;font-weight:600;">
                🎯 Final Output: 1D Probability Logit for Money Laundering Risk
            </div>
        </div>
        '''))

    def plot_confusion_matrix(tn, fp, fn, tp):
        fig = go.Figure(data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=['Predicted Legitimate', 'Predicted Laundering'],
            y=['Actual Legitimate', 'Actual Laundering'],
            colorscale='Blues',
            text=[[f"True Negatives (Safe)<br><b>{tn:,}</b>", f"False Positives (False Alarm)<br><b>{fp:,}</b>"], 
                  [f"False Negatives (Missed)<br><b>{fn:,}</b>", f"True Positives (Caught)<br><b>{tp:,}</b>"]],
            texttemplate="%{text}",
            textfont={"size": 14},
            hoverinfo="none",
            showscale=False
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=320,
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')
        )
        return fig

    with tab_train:
        st.markdown("### 📊 Training Set Performance")
        st.markdown("Metrics evaluated on the massive 7,000,000 transaction training graph.")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Precision", "0.9949")
        m2.metric("Recall", "0.9976")
        m3.metric("F1 Score", "0.9962")
        m4.metric("Accuracy", "0.9966")
        m5.metric("ROC-AUC", "0.9999")
        m6.metric("PR-AUC", "0.9998")

        st.markdown("---")
        st.markdown("#### Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix(3816314, 16198, 7602, 3159886), use_container_width=True)

    with tab_test:
        st.markdown("### 📊 Testing Set Performance")
        st.markdown("Metrics evaluated on the unseen 1,500,000 transaction testing graph.")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Precision", "0.9938")
        m2.metric("Recall", "0.9965")
        m3.metric("F1 Score", "0.9951")
        m4.metric("Accuracy", "0.9955")
        m5.metric("ROC-AUC", "0.9998")
        m6.metric("PR-AUC", "0.9999")

        st.markdown("---")
        st.markdown("#### Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix(801011, 4319, 2431, 692239), use_container_width=True)
"""
    new_content.append(metrics_ui)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_content)
    print('Successfully updated UI.')
else:
    print('Could not find Metrics PAGE marker.')
