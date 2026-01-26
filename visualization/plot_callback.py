import numpy as np

from matplotlib import pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'

from include.dasIT.dasIT.features.signal import logcompression



def plot_mmode(data=None, compression=None, figsize=None):
    if compression:
        data = logcompression(data, compression)
    if figsize:
        fig = plt.figure(figsize=figsize, dpi=300)
    plt.imshow(data, cmap='grey', aspect='auto')
    plt.tight_layout()
    plt.show()


def MmodeSegments(data=None, segments=None):
    data = np.squeeze(data)
    n_segments = len(segments)
    n_cols = max(n_segments, 4)
    fig = make_subplots(
        rows=2, cols=n_cols,
        row_heights=[0.6, 0.4],
        specs=[[{"colspan": n_cols}] + [None]*(n_cols-1)] +
             [[{} for _ in range(n_cols)]],
        vertical_spacing=0.12
    )

    # Top image
    img_fig = px.imshow(data, color_continuous_scale='gray', aspect='auto')
    for trace in img_fig.data:
        fig.add_trace(trace, row=1, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=1)

    # Draw rectangles and label_logic (reverse numbering for the top)
    for idx, (start, end) in enumerate(segments):
        window_label = n_segments - idx
        fig.add_shape(
            type="rect",
            x0=start, x1=end, y0=0, y1=data.shape[0],
            xref="x1", yref="y1",
            fillcolor=f"rgba(30,144,255,0.2)",
            line=dict(width=2, color=f"rgba(30,144,255,1)"),
            row=1, col=1,
            layer="above"
        )
        fig.add_annotation(
            text=f"Window {window_label}",
            x=(start+end)/2,
            y=5,
            xref="x1",
            yref="y1",
            showarrow=False,
            font=dict(size=14, color="black"),
            align="center",
            bgcolor="rgba(255,255,255,0.6)",
            row=1, col=1
        )

    # For the bottom row, show rightmost segment (Window 1) in first (leftmost) subplot
    for plot_idx, seg_idx in enumerate(reversed(range(n_segments))):
        start, end = segments[seg_idx]
        window_label = plot_idx + 1
        seg = data[:, start:end]
        seg_fig = px.imshow(seg, color_continuous_scale='gray', aspect='auto')
        for trace in seg_fig.data:
            fig.add_trace(trace, row=2, col=plot_idx+1)
        fig.update_xaxes(title_text="Scanlines", row=2, col=plot_idx+1)
        fig.update_yaxes(title_text="Samples" if plot_idx==0 else None, showticklabels=(plot_idx==0), row=2, col=plot_idx+1)
        fig.add_annotation(
            text=f"Window {window_label}: {start}-{end}",
            xref=f"x{plot_idx+n_cols+1} domain", yref=f"y{plot_idx+n_cols+1} domain",
            x=0.5, y=1.08, showarrow=False,
            font=dict(size=12), row=2, col=plot_idx+1
        )
        fig.update_yaxes(autorange='reversed', row=2, col=plot_idx+1)

    fig.update_layout(
        title_text='M-mode image and segments',
        height=600,
        width=1200,
        showlegend=False,
        coloraxis_showscale=False
    )
    fig.show()

if __name__ == '__main__':
    m_mode_img = np.random.rand(100, 300)
    segments = [
        (20, 60),
        (80, 120),
        (150, 200),
        (230, 270)
    ]
    MmodeSegments(data=m_mode_img, segments=segments)
