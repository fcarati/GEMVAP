from plotly.graph_objs import Figure, Bar, Layout
from plotly.graph_objs.layout import Margin
from math import log


def ks_barchart(pvalues, scores, title, x_title, alpha=0.05, height=900):
    keys, values = [], []
    for k, v in sorted(scores.items(), reverse=True):
        keys.append(k)
        values.append(pvalues[k])

    data = Bar(
        x=values,
        y=keys,
        orientation='h'
    )
    layout = Layout(
        title=title,
        margin=Margin(
            l=300,
            pad=4
        ),
        xaxis=dict(title=x_title),
        shapes=[{
            'type': 'line',
            'xref': 'x',
            'yref': 'y',
            'x0': alpha,
            'y0': -0.5,
            'x1': alpha,
            'y1': len(pvalues)-0.5,
            'line': {
                'color': 'rgb(250, 171, 96)',
                'width': 3,
            },
        }],
        boxmode='group',
        height=900
    )
    fig = Figure(data=[data], layout=layout)

    return fig
