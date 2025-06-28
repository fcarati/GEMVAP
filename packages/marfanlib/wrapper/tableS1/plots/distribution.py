from plotly.graph_objs import Box, Layout, Figure


def boxplots_groups(pd, label_vector, title, ytitle):
    predictors = pd.columns

    data = []
    for predictor in predictors:
        trace = Box(y=pd[predictor],
                    x=label_vector,
                    name=predictor)
        data.append(trace)

    layout = Layout(title=title,
                    yaxis=dict(title=ytitle, zeroline=False),
                    boxmode='group')

    fig = Figure(data=data, layout=layout)

    return fig
