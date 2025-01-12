import streamlit as st

# Function to read the checkpoint file
def read_checkpoint_file():
    with open('st_checkpoint_2', 'r') as file:
        data = file.read()
    return data

# Display File Contents in Streamlit
st.title("Streamlit App with Checkpoint")
checkpoint_data = read_checkpoint_file('app_checkpoint')
st.write("Contents of checkpoint file:")
st.write(checkpoint_data)

