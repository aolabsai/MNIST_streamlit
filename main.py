import os
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import streamlit_analytics2
from PIL import Image

import data_prep as data

import ao_core as ao
from arch__MNIST import arch_bw, arch_gr


def streamlit_setup():
    if "agent" not in st.session_state:
        st.session_state.agent = setup_agent()
    return

# Throwing a memory problem when consistently switch between app type. Need a better approach to clean up the streamlit session when changing the app type. 
def update_agent():
    del st.session_state.agent
    st.session_state.agent = setup_agent()

def setup_agent():
    if "app_type" not in st.session_state:
        st.session_state.app_type = "Grey scale MNIST"
    if st.session_state.app_type == "Black & White MNIST":
        agent = ao.Agent(arch_bw, notes="B&W MNIST Agent", save_meta=False)
    else:   
        agent = ao.Agent(arch_gr, notes="Default agent", save_meta=False)  # Default for grey scale MNIST
    return agent


def initialize_session_state():
    if "interrupt" not in st.session_state:
        st.session_state.interrupt = False


def reset_interrupt():
    st.session_state.interrupt = False


def set_interrupt():
    st.session_state.interrupt = True


def run_agent(user_STEPS, INPUT, LABEL=[]):
    # running the Agent
    st.session_state.agent.reset_state()
    print(LABEL)
    if np.shape(LABEL)[0] == 0:
        for x in np.arange(user_STEPS):
            print("step: " + str(x))
            # core method to run Agents
            st.session_state.agent.next_state(INPUT, DD=False, unsequenced=True)
    else:
        print("labelled")
        # core method to run Agents
        st.session_state.agent.next_state(INPUT, LABEL, DD=False, unsequenced=True)

    # saving results
    s = st.session_state.agent.state
    q_index = st.session_state.agent.arch.Q__flat
    # z = st.session_state.agent.arch.Z__flat
    z_index = st.session_state.agent.arch.Z__flat
    if st.session_state.app_type == "Black & White MNIST":
        st.session_state.agent_qresponse = np.reshape(
        st.session_state.agent.story[s - 1, q_index], [28, 28]
        )
    else:
         st.session_state.agent_qresponse = np.reshape(
        st.session_state.agent.story[s - 1, q_index], [28, 28,8]
        )   
    # st.session_state.agent_zresponse = st.session_state.agent.story[s, z]
    z = st.session_state.agent.story[s - 1, z_index]

    # return st.session_state.agent_zresponse
    return z


def run_trials(is_training, num_trials, user_STEPS):

    initialize_session_state()

    trialset = data.MN_TRAIN if is_training else data.MN_TEST
    trialset_z = data.MN_TRAIN_Z if is_training else data.MN_TEST_Z

    selected_in, selected_z = data.random_sample(num_trials, trialset, trialset_z)

    # Just training on fonts
    if is_training and (
        num_trials == 0 or "MNIST" not in st.session_state.training_sets
    ):
        selected_in, selected_z = data.select_training_fonts(
            st.session_state.training_sets
        )
    elif is_training and (
        "MNIST" in st.session_state.training_sets
        and len(st.session_state.training_sets) > 1
    ):
        font_in, font_out = data.select_training_fonts(st.session_state.training_sets)
        selected_in = np.append(selected_in, font_in, axis=0)
        selected_z = np.append(selected_z, font_out, axis=0)

    correct_responses = 0
    num_trials = len(selected_in)

    if is_training:
        if st.session_state.app_type == "Black & White MNIST":
            INPUT = data.down_sample(selected_in).reshape(num_trials, 784)
        else:
            INPUT = data.bitmap_to_binary(selected_in).reshape(num_trials, 784*8)    
        st.session_state.agent.next_state_batch(INPUT, selected_z, unsequenced=True)
        print("Training complete; neurons updated.")
        return

    st.session_state.num_trials_actual = 0

    progress_bar = st.progress(float(0))
    for t in np.arange(num_trials):
        nt = t/num_trials
        progress_bar.progress( nt, text="Testing in Progress")

        @st.dialog("Process Interrupted")
        def interrupt_modal_dialog():
            st.warning(
                "Function interrupted! Click the *Re-Enable Processing* button in the sidebar to train/test again."
            )

        if st.session_state.interrupt:
            interrupt_modal_dialog()
            break

        if st.session_state.app_type == "Black & White MNIST":
            INPUT = data.down_sample(selected_in[t, :, :]).reshape(784)
        else:
            INPUT = data.bitmap_to_binary(selected_in[t, :, :]).reshape(784*8)    
        LABEL = selected_z[t]
        if is_training:
            user_STEPS = 1
            run_agent(user_STEPS, INPUT, LABEL)
            print("Trained on " + str(t))
        else:
            response_agent = run_agent(user_STEPS, INPUT, LABEL=[])
            if np.array_equal(response_agent, LABEL):
                correct_responses += 1
            print("Tested on " + str(t))
            print("TOTAL CORRECT-----------------" + str(correct_responses))

        st.session_state.num_trials_actual += 1

        trial_result = (correct_responses / st.session_state.num_trials_actual) * 100
        st.session_state.correct_responses = correct_responses
        st.session_state.trial_result = trial_result
        print("Correct on {x}%".format(x=trial_result))
        # return correct_responses
    progress_bar.empty()


def run_canvas():
    if st.session_state.app_type == "Black & White MNIST":
        input = data.down_sample(st.session_state.canvas_image).reshape(784)
    else:
        input = data.bitmap_to_binary(st.session_state.canvas_image).reshape(784*8)

    label = []
    user_steps = 10
    if st.session_state.train_canvas:
        # label = list(np.binary_repr(int(canvas_label), 4))
        label = data.process_labels_digital(np.array(int(canvas_label))) 
        user_steps = 1
    response = run_agent(user_steps, input, LABEL=label)
    print(response)
    response_int = int("".join(str(x) for x in response), 2)
    st.session_state.canvas_int = response_int
    return


# Used to construct images of agent state
def bin_to_pix(img):
    if img.ndim == 1:
        return np.array(int(''.join(map(str, img)), 2),dtype=np.uint8)
    # elif img.ndim == 2:
    #     return np.array([int(''.join(map(str, pixel)), 2) for pixel in img], dtype=np.uint8)
    else:
        return np.array([bin_to_pix(sub_array) for sub_array in img],dtype=np.uint8)
    
def arr_to_img(img_array, enlarge_factor=15):
    # if else statement below is hard coded and need to be restructured. 
    # Convert the binary array to a numpy array
    if st.session_state.app_type == "Black & White MNIST":
        img_array = np.array(img_array, dtype=np.uint8)

        # Scale the values to 0 or 255 (black or white)
        img_array = img_array * 255

    else:
        if img_array.ndim == 1:
            img_array = np.array(img_array, dtype=np.uint8)
            img_array = img_array * 255

        else:
            # Convert the binary array to a numpy array
            img_array = bin_to_pix(img_array)
            #img_array = img_array.astype(np.uint8)    

    enlarged_array = np.repeat(img_array, enlarge_factor, axis=0)
    try:
        enlarged_array = np.repeat(enlarged_array, enlarge_factor, axis=1)
    except:
        enlarged_array = np.tile(enlarged_array, [enlarge_factor, 1])
        pass

    # Create an image from the array
    img = Image.fromarray(enlarged_array, mode="L")  # 'L' mode is for grayscale

    return img

streamlit_analytics2.start_tracking()
# Basic streamlit setup
st.set_page_config(
    page_title="MNIST Demo by AO Labs",
    page_icon="misc/ao_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://discord.gg/Zg9bHPYss5",
        "Report a bug": "mailto:eng@aolabs.ai",
        "About": "AO Labs builds next-gen AI models that learn after training; learn more at docs.aolabs.ai/docs/mnist-benchmark",
    },
)

streamlit_setup()
# apptype_setup()

st.title("Understanding *Weightless* NNs via MNIST")
st.write("### *a demo by [aolabs.ai](https://www.aolabs.ai/)*")

train_max = 60000
test_max = 10000

############################################################################
with st.sidebar:
    st.write("## Current Active Agent:")
    st.write(st.session_state.agent.notes)

    start_button = st.button(
        "Re-Enable Training & Testing",
        on_click=reset_interrupt,
        help="If you stopped a process\n click to re-enable Testing/Training agents.",
    )
    stop_button = st.button(
        "Stop Testing",
        on_click=set_interrupt,
        help="Click to stop a current Test if it is taking too long.",
    )

    st.write("---")
    st.write("## Load Agent:")

    def load_pickle_files(directory):
        pickle_files = [
            f[:-10] for f in os.listdir(directory) if f.endswith(".ao.pickle")
        ]  # [:-10] is to remove the "ao.pickle" file extension
        return pickle_files

    # directory_option = st.radio(
    #     "Choose directory to retrieve Agents:",
    #     ("App working directory", "Custom directory"),
    #     label_visibility="collapsed"
    # )
    # if directory_option == "App working directory":
    directory = os.path.dirname(os.path.abspath(__file__))
    # else:
    #     directory = st.text_input("Enter a custom directory path:")

    if directory:
        pickle_files = load_pickle_files(directory)

        if pickle_files:
            selected_file = st.selectbox(
                "Choose from saved Agents:", options=pickle_files
            )

            if st.button(f"Load {selected_file}"):
                file_path = os.path.join(directory, selected_file)
                st.session_state.agent = ao.Agent.unpickle(
                    file=file_path, custom_name=selected_file
                )
                st.session_state.agent._update_neuron_data()
                st.write("Agent loaded")
        else:
            st.warning("No Agents saved yet-- be the first!")

    st.write("---")
    st.write("## Save Agent:")

    agent_name = st.text_input(
        "## *Optional* Rename active Agent:", value=st.session_state.agent.notes
    )
    st.session_state.agent.notes = agent_name

    @st.dialog("Save successful!")
    def save_modal_dialog():
        st.write("Agent saved to your local disk (in the same directory as this app).")

    agent_name = agent_name.split("\\")[-1].split(".")[0]
    if st.button("Save " + agent_name):
        st.session_state.agent.pickle(agent_name)
        save_modal_dialog()

    st.write("---")
    st.write("## Download/Upload Agents:")

    @st.dialog("Upload successful!")
    def upload_modal_dialog():
        st.write(
            "Agent uploaded and ready as *Newly Uploaded Agent*, which you can rename during saving."
        )

    uploaded_file = st.file_uploader(
        "Upload .ao.pickle files here", label_visibility="collapsed"
    )
    if uploaded_file is not None:
        if st.button("Confirm Agent Upload"):
            st.session_state.agent = ao.Agent.unpickle(
                uploaded_file, custom_name="Newly Uploaded Agent", upload=True
            )
            st.session_state.agent._update_neuron_data()
            upload_modal_dialog()

    @st.dialog("Download ready")
    def download_modal_dialog(agent_pickle):
        st.write(
            "The Agent's .ao.pickle file will be saved to your default Downloads folder."
        )

        # Create a download button
        st.download_button(
            label="Download Agent: " + st.session_state.agent.notes,
            data=agent_pickle,
            file_name=st.session_state.agent.notes,
            mime="application/octet-stream",
        )

    if st.button("Prepare Active Agent for Download"):
        agent_pickle = st.session_state.agent.pickle(download=True)
        download_modal_dialog(agent_pickle)
############################################################################

agent_col, state_col = st.columns(2)

with agent_col:
    with st.expander("#### Batch Training & Testing", expanded=True):
        st.write("---")
        
        st.selectbox(
                    "Select the application",
                    options=["Grey scale MNIST", "Black & White MNIST"],
                    index=0,
                    help="Select among grey scale and b&w MNIST",
                    key="app_type", 
                    on_change=update_agent  # cal back function 
                    )
        print(st.session_state.app_type)
        

        st.write("##### Training")
        training_set_options = list(data.FONTS.keys())
        training_set_options.insert(0, "MNIST")
        st.session_state.training_sets = st.multiselect(
            "Select training datasets:",
            options=training_set_options,
            default=("MNIST"),
            help="When training on standard fonts (eg. Times New Roman, Arial, etc.), it trains on all of the digits of that font.",
        )

        train_count = st.number_input(
            "Set the number of MNIST training pairs:",
            1,
            train_max,
            value=2,
            help="Randomly selected from MNIST's 60k training set.",
        )
        st.button(
            "Train Agent",
            on_click=run_trials,
            args=(True, train_count, 1),
            disabled=len(st.session_state.training_sets) == 0,
        )
        st.write("---")
        st.write("##### Testing")
        t_count, t_steps = st.columns(2)
        with t_count:
            test_count = st.number_input(
                "Number of test images",
                1,
                test_max,
                value=1,
                help="Randomly selected from MNIST's 10k test set.",
            )
        with t_steps:
            user_STEPS = st.number_input(
                "Number of steps per test image:",
                1,
                20,
                value=10,
                help="10 is a good default; this level of agent usually converges on a stable pattern after ~7 steps (if you've trained it enough).",
            )
        st.button(
            "Test Agent", on_click=run_trials, args=(False, test_count, user_STEPS)
        )

        st.write("---")

        # display trial result
        if "trial_result" in st.session_state:
            st.write("##### Test Results")
            st.write(
                "The agent predicted {correct} out of {total} images correctly, an accuracy of:".format(
                    correct=st.session_state.correct_responses,
                    total=st.session_state.num_trials_actual,
                )
            )
            st.write("# {result}%".format(result=st.session_state.trial_result))

    with st.expander("#### Continuous Learning", expanded=True):
        st.write(
            "You can also train or test your agent on custom inputs made using the canvas below-- try drawing a digit."
        )

        t_canvas, t_label = st.columns(2)
        with t_canvas:
            canvas_result = st_canvas(
                # should probably try to match it to the bg color of MNIST
                background_color="#000000",
                stroke_color="#FFFFFF",
                height=280,
                width=280,
                drawing_mode="freedraw",
                update_streamlit=True,
                key="canvas",
            )

        with t_label:
            st.session_state.train_canvas = st.toggle("Train on canvas")
            canvas_label = st.number_input(
                "Provide a label:", 0, 9, disabled=(not st.session_state.train_canvas)
            )

        if canvas_result.image_data is not None:
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
            input_image_gs = input_image.convert("L")
            resized_gs = input_image_gs.resize((28, 28), Image.Resampling.LANCZOS)
            np_gs = np.array(resized_gs)
            st.session_state.canvas_image = np_gs

        if st.session_state.train_canvas:
            canvas_button_text = "Train on canvas image with label: " + str(
                canvas_label
            )
        else:
            canvas_button_text = "Test on canvas image"

        st.button(canvas_button_text, on_click=run_canvas)

        if "canvas_int" in st.session_state:
            st.write("Drawing identified as:")
            st.write("# {x}".format(x=st.session_state.canvas_int))

with state_col:
    st.write("#### Agent Visual Inspector - view the agent's state history")
    instruction_md = """
    Weightless neural network agents function as *neural state machines*, so during Testing, an agent is shown an image from MNIST and its inner and output states will change in response, allowing you to 'see' what the agent is thinking (unlike deep learning which remains a blackbox); the final output state is translated into an integer label to determine the accuracy of the agent's inference. \n
    You can view all that information by cycle through the states below. \n
    * ***Input*** is the 28x28 B&W pixel input to the agent from MNIST or your canvas (MNIST is grayscale but for this demo we're downsampling to B&W). \n
    * ***Inner State*** is visual representation of 28x28 binary neurons that make up the agent's inner or hidden layer (the same shape as the input, to aid with visual inspection.) \n
    * ***Output State*** is a visual representation of 4 binary neurons (also displayed as a list) that make up the agent's output layer (the states of the binary neurons are translated to an integer label, 0-9). \n
    \n
    Starting from state 1, you'll first cycle through the training data you fed the agent-- you'll notice there's noise interspersed between the training states; this is because we're not tasking the agent with learning a sequence between the MNIST data, so we introduce randomness in between. \n
    When you cycle through to the testing states, you'll see a fixed input with an evolving inner and output states. Often they'll converge on a pattern which correlates with the label of the input image. \n
    """
    with st.expander("About"):
        st.markdown(instruction_md)

    if st.session_state.agent.state - 1 == 0:
        min_value = 0
    else:
        min_value = 1

    sel_state = st.number_input(
        "Displaying state:",
        min_value,
        st.session_state.agent.state,
        value=st.session_state.agent.state - 1,
        help="The agent has history up until state: {}".format(
            st.session_state.agent.state
        ),
    )

    I_col, Q_col, Z_col = st.columns(3)

    with I_col:
        st.write("##### Input")
        i_arr = st.session_state.agent.story[
            sel_state, st.session_state.agent.arch.I__flat
        ]
        if st.session_state.app_type == "Black & White MNIST":
            i_arr = np.reshape(i_arr, [28, 28])
        else:
            i_arr = np.reshape(i_arr, [28, 28,8])    
        i_img = arr_to_img(i_arr)
        st.image(i_img)

    with Q_col:
        st.write("##### Inner State")
        q_arr = st.session_state.agent.story[
            sel_state, st.session_state.agent.arch.Q__flat
        ]
        if st.session_state.app_type == "Black & White MNIST":
            q_arr = np.reshape(q_arr, [28, 28])
        else:   
            q_arr = np.reshape(q_arr, [28, 28,8]) 
        q_img = arr_to_img(q_arr)
        st.image(q_img)

    with Z_col:
        st.write("##### Output State")
        z_arr = st.session_state.agent.story[
            sel_state, st.session_state.agent.arch.Z__flat
        ]
        z_int = z_arr.dot(2 ** np.arange(z_arr.size)[::-1])
        z_img = arr_to_img(z_arr)
        st.write("Result in binary:")
        st.image(z_img)
        st.write("  " + str(z_arr))
        st.write("Result as an integer label: " + str(z_int))

st.write("---")
footer_md = """
    [View & fork the code behind this application here.](https://github.com/aolabsai/MNIST_streamlit) \n
    To learn more about Weightless Neural Networks and the new generation of AI we're developing at AO Labs, [visit our docs.aolabs.ai.](https://docs.aolabs.ai/docs/mnist-benchmark)\n
    \n
    We eagerly welcome contributors and hackers at all levels! [Say hi on our discord.](https://discord.gg/Zg9bHPYss5)
    """
st.markdown(footer_md)
st.image("misc/aolabs-logo-horizontal-full-color-white-text.png", width=300)

streamlit_analytics2.stop_tracking()
