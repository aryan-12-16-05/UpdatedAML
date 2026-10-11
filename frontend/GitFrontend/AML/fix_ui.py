import sys

file_path = 'C:/Users/aryan/frontend/GitFrontend/AML/app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
for i, line in enumerate(lines):
    if '#  MACRO ANALYTICS PAGE' in line:
        start_idx = i - 1
        break

if start_idx != -1:
    new_content = lines[:start_idx]
    
    metrics_ui = """# ══════════════════════════════════════════════════════════════════════════════
#  MODEL METRICS PAGE (Non-Technical UI)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Metrics":
    st.html(textwrap.dedent('''
    <div class="center-card" style="margin-bottom:20px;">
        <h2 style="margin:0;color:#0f172a;font-weight:800;">How the AI Works & Its Accuracy</h2>
        <p style="color:#64748b;font-size:15px;margin-top:5px;">
            A simple breakdown of how our AI detects hidden money laundering rings and how accurate it is.
        </p>
    </div>
    '''))

    st.markdown("### 🧠 How does the AI make decisions?")
    st.markdown('''
    Instead of just looking at a single transaction in isolation, our AI uses a **Graph Neural Network**. 
    This means it looks at the entire "web" of money movement to catch complex laundering patterns like **Fan-Outs** (one person sending money to many) or **Cycles**.
    ''')

    st.html(textwrap.dedent('''
    <div style="display:flex;gap:15px;margin-bottom:25px;flex-wrap:wrap;">
        <div style="flex:1;min-width:250px;background:white;padding:20px;border-radius:12px;border:1px solid #e2e8f0;border-top:4px solid #3b82f6;box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <h4 style="margin-top:0;color:#1e40af;font-size:16px;">1. What it sees (Input)</h4>
            <p style="color:#475569;font-size:14px;">It looks at the sender's history, the receiver's history, and the transaction details (amount, currency, time).</p>
        </div>
        <div style="flex:1;min-width:250px;background:white;padding:20px;border-radius:12px;border:1px solid #e2e8f0;border-top:4px solid #8b5cf6;box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <h4 style="margin-top:0;color:#5b21b6;font-size:16px;">2. How it thinks (The Brain)</h4>
            <p style="color:#475569;font-size:14px;">It connects the dots. If the sender is connected to known suspicious accounts, the AI pays closer attention.</p>
        </div>
        <div style="flex:1;min-width:250px;background:white;padding:20px;border-radius:12px;border:1px solid #e2e8f0;border-top:4px solid #ef4444;box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <h4 style="margin-top:0;color:#991b1b;font-size:16px;">3. What it decides (Output)</h4>
            <p style="color:#475569;font-size:14px;">It gives a final <strong>Risk Score (0% to 100%)</strong>. If the score is very high, it alerts the banking auditors.</p>
        </div>
    </div>
    '''))

    st.markdown("### 📊 How accurate is it?")
    st.markdown("We tested the AI on millions of historical transactions. Here is how well it performs in the real world.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.html('''
        <div style="background:#f0fdf4;padding:20px;border-radius:12px;border:1px solid #bbf7d0;text-align:center;">
            <div style="font-size:32px;font-weight:800;color:#166534;">99.8%</div>
            <div style="font-size:15px;font-weight:600;color:#15803d;margin-top:5px;">Overall Accuracy</div>
            <div style="font-size:13px;color:#166534;margin-top:5px;">The percentage of times the AI was completely correct.</div>
        </div>
        ''')

    with col2:
        st.html('''
        <div style="background:#eff6ff;padding:20px;border-radius:12px;border:1px solid #bfdbfe;text-align:center;">
            <div style="font-size:32px;font-weight:800;color:#1e40af;">99.8%</div>
            <div style="font-size:15px;font-weight:600;color:#1d4ed8;margin-top:5px;">Detection Rate</div>
            <div style="font-size:13px;color:#1e40af;margin-top:5px;">Out of all the actual laundering cases, we successfully caught 99.8%.</div>
        </div>
        ''')

    with col3:
        st.html('''
        <div style="background:#fef2f2;padding:20px;border-radius:12px;border:1px solid #fecaca;text-align:center;">
            <div style="font-size:32px;font-weight:800;color:#991b1b;">Very Low</div>
            <div style="font-size:15px;font-weight:600;color:#b91c1c;margin-top:5px;">False Alarms</div>
            <div style="font-size:13px;color:#991b1b;margin-top:5px;">When the AI flags a transaction, it is almost always genuinely suspicious.</div>
        </div>
        ''')

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Show Detailed Technical Metrics (For Data Scientists)"):
        st.markdown('''
        **Test Set Performance (1.5 Million Transactions)**
        * **Precision:** 0.9988
        * **Recall:** 0.9984
        * **F1 Score:** 0.9986
        * **Accuracy:** 0.9978
        * **ROC-AUC:** 0.9998
        * **PR-AUC:** 0.9999
        ''')
"""
    new_content.append(metrics_ui)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_content)
    print('Successfully updated UI.')
else:
    print('Could not find MACRO ANALYTICS PAGE marker.')
