"""
exporter.py — Report generator for the Streamlit dashboard.
Compiles calibrated parameters, error metrics, and smile plots into PDF and HTML reports.
"""

import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

def generate_pdf_report(model_name, param_df, error_metrics, T_grid, K_grid, market_iv, reconstructed_iv):
    """
    Generates a multi-page PDF calibration report as bytes.
    """
    buffer = io.BytesIO()
    
    # Configure matplotlib styles for clean report layouts
    plt.style.use('default')
    
    with PdfPages(buffer) as pdf:
        # Page 1: Metadata, parameter table, and errors
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        fig.suptitle(f"Volatility Calibration Report: {model_name}", fontsize=18, fontweight='bold', y=0.95, color="#104E8B")
        
        # Timestamp
        fig.text(0.1, 0.88, f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10, style='italic', color="#555")
        
        # Section: Parameters
        fig.text(0.1, 0.82, "Calibrated Parameter Values:", fontsize=14, fontweight='bold', color="#104E8B")
        
        columns = list(param_df.columns)
        rows = [[str(x) if not isinstance(x, float) else f"{x:.6f}" for x in r] for r in param_df.values.tolist()]
        table_data = [columns] + rows
        
        tb = ax.table(cellText=table_data, loc='center', bbox=[0.1, 0.45, 0.8, 0.3])
        tb.auto_set_font_size(False)
        tb.set_fontsize(10)
        
        # Style table headers
        for col_idx in range(len(columns)):
            cell = tb[0, col_idx]
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#104E8B')
            
        # Section: Errors
        fig.text(0.1, 0.35, "Calibration Quality Metrics:", fontsize=14, fontweight='bold', color="#104E8B")
        err_y = 0.30
        for k, v in error_metrics.items():
            val_str = f"{v:.6e}" if isinstance(v, float) else str(v)
            fig.text(0.12, err_y, f"• {k}: {val_str}", fontsize=11)
            err_y -= 0.04
            
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 2: Smile Fits (plots)
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle("Volatility Smile Fit (Selected Maturities)", fontsize=16, fontweight='bold', color="#104E8B", y=0.96)
        
        # Choose 4 indices to represent short, mid, and long maturities
        slice_indices = [0, 2, 4, 7] if len(T_grid) >= 8 else list(range(min(4, len(T_grid))))
        
        for idx, ax_slice in zip(slice_indices, axes.ravel()):
            if idx < len(T_grid):
                T = T_grid[idx]
                ax_slice.plot(K_grid, market_iv[idx, :], 'o-', label="Market", color='#1f77b4', markersize=5)
                ax_slice.plot(K_grid, reconstructed_iv[idx, :], 'x--', label="Fitted", color='#d62728', markersize=5)
                ax_slice.set_title(f"Maturity T = {T:.2f}", fontsize=11, fontweight='bold')
                ax_slice.set_xlabel("Log-Moneyness", fontsize=9)
                ax_slice.set_ylabel("Implied Volatility", fontsize=9)
                ax_slice.legend(fontsize=8)
                ax_slice.grid(True, linestyle=':', alpha=0.6)
            else:
                ax_slice.axis('off')
                
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])
        pdf.savefig(fig)
        plt.close(fig)
        
    buffer.seek(0)
    return buffer.getvalue()

def generate_html_report(model_name, param_df, error_metrics):
    """
    Generates a printable HTML report page.
    """
    param_table_html = param_df.to_html(classes="param-table", index=False)
    
    metrics_html = ""
    for k, v in error_metrics.items():
        val_str = f"{v:.6e}" if isinstance(v, float) else str(v)
        metrics_html += f'<div class="metric"><strong>{k}:</strong> {val_str}</div>'
        
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Calibration Report - {model_name}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 40px;
                color: #333;
                line-height: 1.5;
            }}
            .header {{
                border-bottom: 3px solid #104E8B;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                color: #104E8B;
                margin: 0 0 10px 0;
                font-size: 24px;
            }}
            .header .meta {{
                font-size: 14px;
                color: #666;
            }}
            h2 {{
                color: #104E8B;
                font-size: 18px;
                margin-top: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
            }}
            .metric {{
                margin: 8px 0;
                font-size: 15px;
            }}
            .param-table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            .param-table th, .param-table td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: left;
                font-size: 14px;
            }}
            .param-table th {{
                background-color: #104E8B;
                color: white;
                font-weight: bold;
            }}
            .param-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            @media print {{
                body {{
                    margin: 20px;
                }}
                .no-print {{
                    display: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Volatility Model Zoo Calibration Report</h1>
            <div class="meta">
                <strong>Model:</strong> {model_name} <br>
                <strong>Timestamp:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <h2>Calibrated Parameter Results</h2>
        {param_table_html}
        
        <h2>Performance & Error Metrics</h2>
        {metrics_html}
    </body>
    </html>
    """
    return html
