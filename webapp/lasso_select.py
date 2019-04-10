from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button
from bokeh.layouts import column

# setup plot
tools = "pan,wheel_zoom,lasso_select,reset"
fig = figure(title='Select points',
            plot_width=300, plot_height=200,tools=tools)

import numpy as np
x = np.linspace(0,10,100)
y = np.random.random(100) + x

import pandas as pd
data = pd.DataFrame(dict(x=x, y=y))

# define data source
src = ColumnDataSource(data)
# define plot
fig.circle(x='x', y='y', source=src)

# define interaction
def print_datapoints():
    indices = src.selected.indices
    results = data.iloc[indices]
    print(results)
    resultsDict=results.to_dict()['x']
    resultString=str(resultsDict)


btn = Button(label='Selected points', button_type='success')
btn.on_click(print_datapoints)

curdoc().add_root(column(btn,fig))
