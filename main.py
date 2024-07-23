# Import Library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

#file_path = 'F:/Project/Fullstack/Streamlit/SOC Estimation/data/dataset_discharge_clean.csv'
file_path = 'https://raw.githubusercontent.com/wilywho/State-of-Charge-Estimation-App/master/data/dataset_discharge_clean.csv'

@st.cache_data
def load_data(file_path):
    # Read the dataset
    df = pd.read_csv(file_path)
    return df
@st.cache_data
def get_cycle_data(df, cycle):
    return df[df['Cycle'] == cycle]

# Function to create and compile the model
def create_model(input_shape, num_neurons_layer1, num_neurons_layer2):
    model = Sequential()
    model.add(Dense(num_neurons_layer1, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(num_neurons_layer2, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
    return model

# Custom callback to track training time and progress
class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        st.session_state.start_time = time.time()
        st.write(f"Epoch {epoch+1} started at {time.strftime('%X')}")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - st.session_state.start_time
        st.write(f"Epoch {epoch+1} ended at {time.strftime('%X')}, took {elapsed_time:.2f} seconds")
        st.write(f"Loss: {logs['loss']:.4f}, MAE: {logs['mae']:.4f}, Val Loss: {logs['val_loss']:.4f}, Val MAE: {logs['val_mae']:.4f}")

# Title
with header:
    st.title("Data Science Project")
    st.markdown("##### Laboratory of Energy Mangement Bandung Institute of Technology")
    st.markdown("###### Research Engineer: Farhan Wily B.Sc.")
    st.header('State of Charge (SOC) Estimation For Lithium-Ion Battery ZTEZXDC48')

with dataset:
    st.header('ZXDC48 LFP Battery Cycle Test Dataset')
    st.text('This dataset contains cycle test data for the ZTE ZXDC48 LFP battery, which has a capacity of 100 Ah and operates at a nominal voltage of 48 V.')
    # Read the dataset
    df = pd.read_csv(file_path)
    st.write(df.head(10))
    
    # Data Visualization
    cycles = df['Cycle'].unique()
    st.header('Voltage vs Time')
    st.markdown('#### Interactive Plot using Plotly in Streamlit')
    
    # Create a plotly figure for all cycles
    fig1 = px.line(df, x='Time (s)', y='Voltage (V)', color='Cycle', title='Voltage vs Time for each Cycle')

    # Update the y-axis range
    fig1.update_yaxes(range=[42, 55])
    
    st.plotly_chart(fig1)

    # SOC Plot
    st.header('Voltage vs SOC Reference')
    fig2 = px.line(df, x='SOC (%)', y='Voltage (V)', color='Cycle', title='Voltage vs SOC Reference for each Cycle')
    
    # Update the y-axis range
    fig2.update_yaxes(range=[43, 53])
    
    st.plotly_chart(fig2)

    st.text('This SOC Reference calculate by using Coulomb Counting Method')
    st.markdown('##### State of Charge (SOC) Formula')
    st.latex(r'''SOC(t) = SOC(t_0) + \frac{1}{C} \sum_{i=0}^{n} I_i \Delta t''')
    

with features:
    st.header("Dataset Features")
    st.write("Below are the features in the dataset:")
    st.table(df.columns)


with model_training:
    st.header('Deep Learning Model using Deep Neural Network Algorithm')
    st.text('In this section, you can select the hyperparameters of the model and observe its performance')

    sel_col, disp_col = st.columns(2)

    # Feature Selection
    available_features = df.columns.tolist()
    input_feature = sel_col.multiselect('Select the input Features: ', available_features, default=['Voltage (V)'])
    st.write(f'Selected features: {input_feature}')

    st.markdown("### Deep Neural Network Hyperparameter Tuning")

    # The number of neuron
    num_neurons_layer1 = sel_col.slider('Number of neurons in Layer 1:', min_value=1, max_value=512, value=128, step=1)
    num_neurons_layer2 = sel_col.slider('Number of neurons in Layer 2:', min_value=1, max_value=512, value=64, step=1)

    # The number of epoch
    max_epoch = sel_col.slider('The number of epochs: ', min_value=5, max_value=100, value = 5, step=5)
    
    # The number of batch size
    n_batch_size = sel_col.selectbox('The number of Batch Size: ', options=[16, 32, 64, 128, 256, 512], index=1)

    # Placeholder untuk mengimplementasikan model training
    st.write("Model training with the following settings:")
    st.write(f"Input features: {input_feature}")
    st.write(f"Layer 1 neurons: {num_neurons_layer1}")
    st.write(f"Layer 2 neurons: {num_neurons_layer2}")
    st.write(f"Epochs: {max_epoch}")
    st.write(f"Batch size: {n_batch_size}")

    # Deep Learning Training

    st.markdown("### Train the Deep Learning Model for SOC Estimation")
    if st.button('Train Model'):
        # Prepare the data
        X = df[input_feature]
        X_values = np.array(X)
        y = df['SOC (%)']
        y_values = np.array(y).reshape(-1,1)

        # Standardize the features
        scaler = StandardScaler()
        X_values_scaled = scaler.fit_transform(X_values)

        # Split the data into training and test sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_values_scaled, y_values, test_size=0.2, random_state=42)

        # Create the model
        model = create_model(X_train.shape[1], num_neurons_layer1, num_neurons_layer2)

        # Create epoch display placeholder
        epoch_display = st.empty()

        # Train the model with the Streamlit callback
        history = model.fit(X_train, y_train, epochs=max_epoch, batch_size=n_batch_size, validation_data=(X_test, y_test),
                            callbacks=[StreamlitCallback()])

        # Train the model
        #history = model.fit(X_train, y_train, epochs=max_epoch, batch_size=n_batch_size, validation_data=(X_test, y_test))

        # Display training progress
        st.write('Model training completed.')
        st.write('Training History:')
        st.write(pd.DataFrame(history.history))

        # Plot training history
        fig,ax=plt.subplots(nrows=1,ncols=3, figsize=(14, 6))
        metric=['mse','mae','mape']
        for i in range(3):
            ax[i].plot(history.history['{}'.format(metric[i])])
            ax[i].plot(history.history['val_{}'.format(metric[i])])
            ax[i].set_title('Model {}'.format(metric[i]), fontsize = 22)
            ax[i].set_ylabel('{}'.format(metric[i]), fontsize = 20)
            ax[i].set_xlabel('epoch', fontsize = 20)
            ax[i].legend(['train', 'validation'], loc='upper left')
        
        plt.tight_layout()
        st.pyplot(fig)

        # Predict the test set
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Prediction with Model
        SOC_val = model.predict(X_values_scaled)
        df['Estimation SOC (%)'] = SOC_val

        # Plot using Plotly
        #fig = px.scatter(df, x='SOC (%)', y='Voltage (V)', color='Cycle', title='Voltage vs SOC with Estimated SOC')
        #fig.add_scatter(x=df['Estimation SOC (%)'], y=df['Voltage (V)'], mode='markers', name='Estimated SOC', marker=dict(color='red', symbol='x', size=3))
        #fig.update_layout(xaxis_title='SOC (%)', yaxis_title='Voltage (V)')
        #st.plotly_chart(fig)

        # Plot using Matplotlib
        # Create a scatter plot for SOC vs Voltage
        st.markdown("#### Estimated Result")
        plt.figure(figsize=(10, 6))

        # Scatter plot for actual data colored by cycle
        for cycle, group in df.groupby('Cycle'):
            plt.scatter(group['SOC (%)'], group['Voltage (V)'], s=10, label=f'Cycle {cycle}')

        # Scatter plot for estimated SOC with red 'x' markers
        plt.scatter(df['Estimation SOC (%)'], df['Voltage (V)'], color='red', marker='x', s=10, label='Estimated SOC')
        
        # Set labels and title
        plt.xlabel('SOC (%)')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage vs SOC with Estimated SOC')
        plt.legend()
        plt.grid()

        # Show plot
        st.pyplot(plt)

        # Display performance metrics
        st.subheader('Model Performance Summary')
        st.write(f'R²: {r2:.4f}')
        st.write(f'RMSE: {rmse:.4f}')
        st.write(f'MAE: {mae:.4f}')
        st.write(f'MAPE: {mape:.4f}%')

if st.button('Refresh'):
    st.write("Halaman diperbarui pada:", datetime.now())


