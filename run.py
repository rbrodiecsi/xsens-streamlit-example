import streamlit as st
import pandas as pd
import numpy as np

# define columns and column labels for incoming data
used_columns = ['Acc_X','Acc_Y','Acc_Z','Gyr_X','Gyr_Y','Gyr_Z','Mag_X','Mag_Y','Mag_Z']
labels = [f'{sensor} ({axis} axis)' for sensor in ['Acceleration','Angular Rate','Magnetometer'] for axis in ['X','Y','Z']]

# maximum file length for incoming data will be 2 minutes
# Download and modify this code to change this and experiment with longer files
MAX_PROCESSING_LENGTH = 120

# normalized cycle length will be 100 samples
NORM_CYCLE_LENGTH = 100

class WorkflowStages():
    """
    State management for workflow
    """
    ERROR = -999
    UPLOAD = 0
    CLIP = 1
    FILTER = 2
    EVENT_DETECT = 3
    OUTPUT_METRICS = 4

def workflow_stage(new_stage=None):
    """

    :param new_stage: New state for the current workflow
    :return:
    """

    if 'workflow_stage' not in st.session_state:
        st.session_state['workflow_stage'] = WorkflowStages.UPLOAD

    if new_stage:
        st.session_state['workflow_stage'] = new_stage

    return st.session_state['workflow_stage']

def clip_data(data, t0, t1, column='Time'):
    """
    Convenience function for clipping data based on

    :param data:
    :param t0:
    :param t1:
    :return:
    """
    return data.loc[(data[column] >= t0) & (data[column] <= t1)]

def calculate_sample_rate(data):
    """
    Calculate collected sample rate from XSens data. This will require that incoming
    data have the *SampleTimeFine* column defined.

    :param data:    pandas.DataFrame with SampleTimeFine defined
    :return:        float of sample rate in samples per second (Hz)
    """
    period = data['SampleTimeFine'].iloc[1:10].diff().mean()

    sample_rate = 1 / (period / 1000000)

    return round(sample_rate, 1)


@st.cache
def load_data(fn):
    """

    Returns a dataset with a sample rate for a given CSV file, formatted as
    downloaded from the XSens Dot software. This is cached by streamlit to facilitate
    repeated processing.

    :param fn:  Uploaded CSV input file
    :return:    pandas.DataFrame containing uploaded data
    """
    df = pd.read_csv(fn)

    return df

def plot_preview(data, x0=None, x1=None, column=None, xcolumn='Time', events=[], title="Data Preview"):
    """
    Produce a timeseries preview of a dataset

    :param data:    Timeseries data
    :param x0:      View start point on the x-axis
    :param x1:      View endpoint on the x-axis
    :param column:  Data column in :data: that will be displayed
    :param events:  A list of cycle events, plotted as vertical lines
    :param title:   Chart title
    :return:        None
    """
    from bokeh.models import Span
    from bokeh.plotting import figure

    display_data = data.loc[(data['Time'] >= x0) & (data['Time'] <= x1)]

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
    """
    Plot cycle data as produced by the extract_cycles function. This will plot each column passed in as a mean+/-SD
    of the signal over the course of the average cycle.

    :param cycles:  dict of column, MxN numpy arrays where each element array is M stacked cycles, data normalized to N
                    samples in length.
    :return:
    """


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


    grid = gridplot([[p] for p in plots],sizing_mode='scale_both')

    st.bokeh_chart(grid, use_container_width=True)

def extract_cycles(data, columns, events):
    """
    Extract and stack cycle data from a signal based on a list of event indexes

    :param data:    Data to break into cycles
    :param columns: Columns to apply this process to
    :param events:  List of series indices where cycle events occur.
    :return:        dict of column, MxN numpy arrays where each element array is M stacked cycles, data normalized to N
                    samples in length.
    """

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

def filter_data(data, filter_type, sample_rate, filter_order, cutoff_frequency ):
    """
    Perform a low-pass butterworth filter of an incoming dataset

    :param data:                Data to be filtered
    :param filter_type:         type of filter from ['lowpass','highpass']
    :param sample_rate:         Sample rate of incomgng data
    :param filter_order:        Filter order to apply
    :param cutoff_frequency:    Cutoff frequency for low-pass filter
    :return:
    """
    from scipy import signal

    sos = signal.butter(filter_order, Wn=cutoff_frequency, btype=filter_type, fs=sample_rate, output='sos')

    output = signal.sosfiltfilt(sos, data)

    return output

def detect_peaks(data, positive=True, peak_distance=None, peak_height=None):
    """
    Perform simple peak detection using scipy.signal.find_peaks

    :param data:            Incoming time-series data
    :param positive:        Find positive peaks if True, negative peaks if False
    :param peak_distance:   Minimum peak distance
    :param peak_height:     Minimum peak height
    :return:                List of peak locations as array indices
    """
    from scipy.signal import find_peaks

    if positive:
        peaks,_ = find_peaks(data, distance=peak_distance, height=peak_height)
    else:
        peaks, _ = find_peaks(-1*data, distance=peak_distance, height=peak_height)

    return peaks

def detect_crossing(data, threshold, ascending=False):
    """
    Detect signal threshold crossings in timeseries data relative to a specified threshold value

    :param data:        Incoming time series data as an array or Series object
    :param threshold:   Threshold value to detect crossing of in signal
    :param ascending:   If True, only crossings approaching from below the threshold will be returned, otherwise
                        only crossing approaching from above threshold will be returned
    :return:            List of array indices meeting the crossing requirements
    """

    crossings = np.diff(data > threshold, prepend=False)
    idx = np.arange(len(data))[crossings]

    if ascending:
        idx = [i for i in idx if data[i-1]<threshold]
    else:
        idx = [i for i in idx if data[i - 1] > threshold]

    return idx

def app():
    """
    Main application script

    :return:
    """

    sample_rate = None
    working_data = None

    st.header("Example workflow")
    st.subheader("Sensor: XSens Dot")

    with st.expander("Upload", expanded=True):
        uploaded_file = st.file_uploader(label="Upload Files", accept_multiple_files=False)
        st.markdown("<p>A prepared ouptut file is available <a href='https://raw.githubusercontent.com/rbrodiecsi/xsens-streamlit-example/main/example_data/xsens_running_data.csv' target='_blank'>HERE</a>. Right click and select <strong>Save Link As...</strong></p>",
                    unsafe_allow_html=True)

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
        with st.expander("Basic Butterworth Filter", expanded=True):

            filter_type = st.selectbox(label="Filter Type",
                                       options=['Low-pass','High-pass'],
                                       index=0)
            filter_type = 'lowpass' if 'Low' in filter_type else 'highpass'

            order_options = ['1st Order','2nd Order','3rd Order']+[f'{o}th Order' for o in range(4,11)]
            order = st.selectbox(label="Filter Order",
                                 options=order_options,
                                 index=1)
            order = order_options.index(order)+1
            cutoff = st.slider("Cutoff Frequency (Hz)",min_value=0.5,max_value=float(sample_rate),step=0.5,value=10.0)

            filtered_data = filter_data(working_data[column],filter_type,sample_rate, order, cutoff)
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

                events = detect_crossing(working_data[column].values,
                                      threshold=threshold,
                                      ascending=ascending)

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

# Run the main streamlit process
app()