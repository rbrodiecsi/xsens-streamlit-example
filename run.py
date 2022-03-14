import streamlit as st
import pandas as pd
import numpy as np

used_columns = ['Acc_X','Acc_Y','Acc_Z','Gyr_X','Gyr_Y','Gyr_Z','Mag_X','Mag_Y','Mag_Z']
labels = [f'{sensor} ({axis} axis)' for sensor in ['Acceleration','Angular Rate','Magnetometer'] for axis in ['X','Y','Z']]

MAX_PROCESSING_LENGTH = 120
NORM_CYCLE_LENGTH = 100

class WorkflowStages():
    ERROR = -999
    UPLOAD = 0
    CLIP = 1
    FILTER = 2
    EVENT_DETECT = 3
    OUTPUT_METRICS = 4

def workflow_stage(new_stage=None):
    if 'workflow_stage' not in st.session_state:
        st.session_state['workflow_stage'] = WorkflowStages.UPLOAD

    if new_stage:
        st.session_state['workflow_stage'] = new_stage

    return st.session_state['workflow_stage']

def clip_data(data, t0, t1):
    return data.loc[(data['Time'] >= t0) & (data['Time'] <= t1)]

def calculate_sample_rate(data):
    period = data['SampleTimeFine'].iloc[1:10].diff().mean()

    sample_rate = 1 / (period / 1000000)

    return round(sample_rate, 1)


@st.cache
def load_data(fn):
    """

    Returns a dataset with a sample rate for a given CSV file, formatted as downloaded from the XSens Dot software.

    :param fn: uploaded CSV input file
    :return:
    """
    df = pd.read_csv(fn)

    return df



def plot_preview(data, x0=None, x1=None, column=None, events=[], title="Data Preview"):
    import random, string
    from bokeh.models import Span
    from bokeh.plotting import figure

    key = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

    display_data = data.loc[(data['Time'] >= x0) & (data['Time'] <= x1)]

    # # show a subsampled dataset
    # if len(display_data) > 5000:
    #     display_data = display_data.sample(min( len(data),5000 ) ).sort_values('Time')

    x = display_data['Time']
    y = display_data[column]

    p = figure(
        plot_height=300,
        title=f'{title} for {column}',
        x_axis_label='Time (s)',
        y_axis_label=labels[used_columns.index(column)])

    p.line(x, y, legend_label=labels[used_columns.index(column)], line_width=0.5)

    for e in events:
        span = Span(location=x.iloc[e],
                    dimension='height',
                    line_color='#000000',
                    line_width=0.25)
        p.renderers.append(span)

    st.bokeh_chart(p, use_container_width=True)

def plot_normalized_cycles(cycles):
    from bokeh.layouts import gridplot
    from bokeh.plotting import figure

    plots = []
    range_obj = None
    for column in cycles:


        if range_obj is None:
            p = figure(x_axis_label='% Cycle', plot_height=150, y_axis_label=labels[used_columns.index(column)])
            range_obj = p.x_range
        else:
            p = figure(x_axis_label='% Cycle', plot_height=150, x_range=range_obj, y_axis_label=labels[used_columns.index(column)])


        plots.append(p)

        val_mean = np.nanmean(cycles[column], axis=0)
        val_sd = np.nanstd(cycles[column], axis=0)

        xdata = np.linspace(0,100,NORM_CYCLE_LENGTH)

        p.line(xdata, val_mean,
               legend_label=labels[used_columns.index(column)],
               line_width=1, color='#000000')

        p.line(xdata, val_mean+val_sd,
               line_dash='dashed',
               legend_label=labels[used_columns.index(column)],
               line_width=1, color='#000000')

        p.line(xdata, val_mean-val_sd,
               line_dash='dashed',
               legend_label=labels[used_columns.index(column)],
               line_width=1, color='#000000')



    # n = 3
    # plots = [plots[i:i + n] for i in range(0, len(plots), n)]

    grid = gridplot([[p] for p in plots],sizing_mode='scale_both')

    st.bokeh_chart(grid, use_container_width=True)

def extract_cycles(data, columns, events):

    raw_cycles = {}
    norm_cycles = {}


    for column in columns:
        raw_cycles[column] = []
        for i in range(len(events) - 1):
            cycle = data[column].iloc[events[i]:events[i + 1]].values
            raw_cycles[column].append(cycle)

        norm_cycles[column] = np.zeros((len(raw_cycles[column]), NORM_CYCLE_LENGTH)) * np.nan
        for i, cycle in enumerate(raw_cycles[column]):  # explain enumerate
            new_index = np.linspace(0, len(cycle) - 1, NORM_CYCLE_LENGTH)
            norm_cycles[column][i, :] = np.interp(new_index, np.arange(len(cycle)), cycle)

    return raw_cycles, norm_cycles

def filter_data(data, sample_rate, filter_order, cutoff_frequency ):
    from scipy import signal

    sos = signal.butter(filter_order, Wn=cutoff_frequency, btype='lowpass', fs=sample_rate, output='sos')

    output = signal.sosfiltfilt(sos, data)

    return output

def detect_peaks(data, positive=True, peak_distance=None, peak_height=None):
    from scipy.signal import find_peaks

    if positive:
        peaks,_ = find_peaks(data, distance=peak_distance, height=peak_height)
    else:
        peaks, _ = find_peaks(-1*data, distance=peak_distance, height=peak_height)

    return peaks

def detect_crossing(data, threshold, ascending=False):
    pass

def app():

    sample_rate = None
    working_data = None

    st.header("Example workflow")
    st.subheader("XSens Dot")

    with st.expander("Upload", expanded=True):
        uploaded_file = st.file_uploader(label="Upload Files", accept_multiple_files=False)

    if uploaded_file:
        if workflow_stage() == WorkflowStages.UPLOAD:
            workflow_stage(WorkflowStages.CLIP)
    else:
        workflow_stage(WorkflowStages.UPLOAD)

    if workflow_stage() >= WorkflowStages.CLIP:
        with st.expander("Clip Region", expanded=True):
            if uploaded_file:

                uploaded_data = load_data(uploaded_file)

                working_data = uploaded_data.copy() # keep caching happy?
                sample_rate = calculate_sample_rate(working_data)
                working_data.insert(0, 'Time', working_data['PacketCounter'] / sample_rate)

                column = st.selectbox(label="Column", options=used_columns, index=0)
                x0, x1 = st.slider("Clip Data",
                       min_value=float(working_data['Time'].min()),
                       max_value=float(working_data['Time'].max()),
                       value=[float(working_data['Time'].min()), float(working_data['Time'].max())])

                plot_preview(working_data, x0, x1, column)
                working_data = clip_data(working_data, x0, x1)

                do_clip = st.button("Accept Clipped Region")
                if do_clip:
                    workflow_stage(WorkflowStages.FILTER)


    if workflow_stage() >= WorkflowStages.FILTER:
        with st.expander("Basic Lowpass Filter", expanded=True):

            order_options = ['1st Order','2nd Order','3rd Order']+[f'{o}th Order' for o in range(4,11)]
            order = st.selectbox(label="Filter Order",
                                 options=order_options,
                                 index=1)
            order = order_options.index(order)+1
            cutoff = st.slider("Cutoff Frequency (Hz)",min_value=0.5,max_value=float(sample_rate),step=0.5,value=10.0)

            filtered_data = filter_data(working_data[column],sample_rate, order, cutoff)
            working_data.loc[:,column]=filtered_data


            plot_preview(working_data, x0, x1, column, title="Filtered Data")

            do_filter = st.button("Accept Filter Settings")
            if do_filter:
                workflow_stage(WorkflowStages.EVENT_DETECT)

    if workflow_stage() >= WorkflowStages.EVENT_DETECT:
        with st.expander("Detect Cycle Events", expanded=True):
            events = []
            event_type = st.selectbox(label="Event Type",
                                      options=["Positive Peak",
                                               "Negative Peak",
                                               "Positive Threshold Crossing",
                                               "Negative Threshold Crossing"],
                                      index=0)
            if "Threshold" in event_type:
                threshold = st.number_input(label="Threshold",
                                min_value=float(working_data[column].min()),
                                max_value=float(working_data[column].max()),
                                value=float(working_data[column].mean()))
                ascending = "Positive" in event_type
            elif "Peak" in event_type:
                # min-height
                min_height = st.number_input(label="Minimum Peak Height",
                                min_value=float(working_data[column].min()),
                                max_value=float(working_data[column].max()-working_data[column].min()),
                                value=float(working_data[column].mean()))

                # min-distance
                min_distance = st.number_input(label="Peak Distance",
                                               min_value=0.0,
                                               max_value=10.0, value=0.2)
                min_distance = min_distance * sample_rate

                # threshold
                threshold = st.number_input(label="Threshold",
                                            min_value=float(working_data[column].min()),
                                            max_value=float(working_data[column].max()),
                                            value=float(working_data[column].mean()))

                positive = "Positive" in event_type

                events = detect_peaks(working_data[column].values,
                                      peak_distance=min_distance,
                                   positive=positive)

                plot_preview(working_data, x0, x1, column, events=events, title="Filtered Data")

            do_metrics = st.button("Accept Detection Settings")
            if do_metrics:
                workflow_stage(WorkflowStages.OUTPUT_METRICS)

    if workflow_stage() >= WorkflowStages.OUTPUT_METRICS:
        with st.expander("Cycle Metrics", expanded=True):
            selected = st.multiselect(label="Signals to Include",options=used_columns,default=column)
            apply_filter = st.checkbox(label="Apply filter settings to signals")

            raw,norm = extract_cycles(working_data,selected,events)

            plot_normalized_cycles(norm)


    # with st.expander("Detect Cycles", expanded=file_is_uploaded):
    #     if uploaded_file:
    #         st.write(uploaded_file.name)
    #     else:
    #         st.write("Upload a file to begin processing...")
    #
    # with st.expander("Output Metrics", expanded=file_is_uploaded):
    #     if uploaded_file:
    #         st.write(uploaded_file.name)
    #     else:
    #         st.write("Upload a file to begin processing...")

app()