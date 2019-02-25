
import numpy as np
import pandas as pd
from bokeh.io import show, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool, Legend
from bokeh.plotting import figure, output_file
from bokeh.palettes import Category20, GnBu3, OrRd3


# convert events to numerical
def convert_events(pca):

    # create new columns
    pca['Procedure Num'] = 0
    pca['Predicted Num'] = 0

    # for each event
    for i, e in enumerate(np.unique(pca['Procedure'])):
        idx1 = np.where(pca['Procedure'] == e)
        idx2 = np.where(pca['predicted'] == e)
        pca['Procedure Num'].iloc[idx1[0]] = i
        pca['Predicted Num'].iloc[idx2[0]] = i

    return pca


# count correctly predicted
def count_predicted(pca):

    # sum up correctly predicted
    predicted = {}
    predicted['Incorrectly'] = []
    predicted['Correctly'] = []
    procedures = [0] * len(np.unique(pca['Procedure']))
    for i, e in enumerate(np.unique(pca['Procedure Num'])):
        idx = np.where(pca['Procedure Num'] == i)
        correct = np.where(pca['Procedure'].iloc[idx[0]] == pca['predicted'].iloc[idx[0]])
        incorrect = np.where(pca['Procedure'].iloc[idx[0]] != pca['predicted'].iloc[idx[0]])
        procedure = pca['Procedure'].iloc[idx[0][0]]
        procedures[e] = procedure
        predicted['Correctly'].append(len(correct[0]))
        predicted['Incorrectly'].append(len(incorrect[0]) * -1)

        # find category inncorrect are
        # incorrectly[procedure] = []
        # for j, e2 in enumerate(np.unique(pca['Procedure Num'])):
            # incorrect = np.where(pca['Predicted Num'].iloc[idx[0]] == e2)
            # incorrectly[procedure].append(len(incorrect[0]) * -1)

    predicted['procedures'] = procedures

    return procedures, predicted


# events prediction plot
def categories_plot(procedures, predicted):

    # create figure
    p = figure(y_range=procedures, plot_height=400, x_range=(min(predicted['Incorrectly']), max(predicted['Correctly'])), \
        title="Predictions", plot_width=1000, toolbar_location=None)
    p.hbar(y=predicted['procedures'], right=predicted['Correctly'], height=0.9, color='Green', legend='Correct')
    p.hbar(y=predicted['procedures'], right=predicted['Incorrectly'], height=0.9, color='Red', legend='Incorrect')

    p.y_range.range_padding = 0.1
    p.ygrid.grid_line_color = None
    p.legend.location = "top_right"
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None

    return p


# timeseries plot
def timeseries_plot(pca):

    # create main plot
    secs = np.array(pca['seconds'], dtype=np.float)
    source = ColumnDataSource(data=dict(secs=secs, close=pca['pca 1']))
    p = figure(plot_height=400, plot_width=1000, tools="", toolbar_location=None,
               x_axis_type="linear", x_axis_location="above", title="Timeseries PCA",
               background_fill_color="#efefef", x_range=(min(secs), max(secs)))
    p.yaxis.axis_label = 'PCA 1'

    # make lines for each sample
    colors = Category20[20]
    plots = [0] * len(np.unique(pca['Sample #']))
    legend_dict = {}
    for i in np.unique(pca['Sample #']):
        idx = np.where(pca['Sample #'] == i)
        x = pca['seconds'].iloc[idx[0]]
        y = pca['pca 1'].iloc[idx[0]]
        plots[i] = p.line(x, y, color=colors[int(pca['Procedure Num'].iloc[idx[0][0]])], alpha=0.5)
        if pca['Procedure'].iloc[idx[0][0]] in legend_dict.keys():
            legend_dict[pca['Procedure'].iloc[idx[0][0]]].append(plots[i])
        else:
            legend_dict[pca['Procedure'].iloc[idx[0][0]]] = [plots[i]]

    # create legend
    legend_list = []
    for d in legend_dict:
        legend_list.append((d, legend_dict[d]))
    legend = Legend(items=legend_list, location=(20, -100))
    p.add_layout(legend, 'right')

    # create bottom scrolling image
    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    plot_height=130, plot_width=1000, y_range=p.y_range,
                    x_axis_type="linear", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")
    range_tool = RangeTool(x_range=p.x_range)
    select.xaxis.axis_label = 'Seconds'
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.scatter('secs', 'close', source=source, line_width=0.25)
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool

    return p, select


# main function
def main():

    # read in ML output
    path = '/Users/deirdre/Documents/VA-ML/Project/EMS-Prediction/'
    data = pd.read_csv(path + 'data/' + 'AllData-RawXY.csv', header=0, sep=',', index_col=0)
    pca = pd.read_csv(path + 'data/' + 'PCA-RawXY.csv', header=0, sep=',', index_col=0)

    # make categories numerical
    pca = convert_events(pca)

    # plot data
    p1, select1 = timeseries_plot(pca)
    procedures, predicted = count_predicted(pca)
    p2 = categories_plot(procedures, predicted)

    # output images
    output_file("timeseries.html", title="Timeseries PCA")
    show(column(p1, select1, p2))


main()
