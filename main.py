import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import data_prep as data

import ao_core as ao
from arch_MNIST import arch


def streamlit_setup():
    if "agent" not in st.session_state:
        st.session_state.agent = setup_agent()


def setup_agent():
    agent = ao.Agent(arch)
    return agent


def run_agent(user_STEPS, INPUT, LABEL=[]):
    # running the Agent
    st.session_state.agent.reset_state()
    print(LABEL)
    if np.shape(LABEL)[0] == 0:
        for x in np.arange(user_STEPS):
            print("NO LABEL")
            # core method to run Agents
            st.session_state.agent.next_state(INPUT, DD=False)
    else:
        print("labelled")
        # core method to run Agents
        st.session_state.agent.next_state(INPUT, LABEL, DD=False)

    # saving results
    s = st.session_state.agent.state
    q = st.session_state.agent.arch.Q__flat
    # z = st.session_state.agent.arch.Z__flat
    z_index = st.session_state.agent.arch.Z__flat
    st.session_state.agent_qresponse = np.reshape(
        st.session_state.agent.story[s - 1, q], [28, 28]
    )
    # st.session_state.agent_zresponse = st.session_state.agent.story[s, z]
    z = st.session_state.agent.story[s - 1, z_index]

    # return st.session_state.agent_zresponse
    return z


def run_trials(is_training, num_trials, user_STEPS):
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
        # print(font_in)
        # print(font_in)
        selected_in = np.append(selected_in, font_in, axis=0)
        selected_z = np.append(selected_z, font_out, axis=0)
        # selected_z.extend(font_out)

    correct_responses = 0
    # print(len(selected_in))
    num_trials = len(selected_in)

     # Being used to test next_state_batch
    if is_training:
        # func = np.vectorize(lambda x: 1 if x >= 200 else 0)
        # func = np.vectorize(data.down_sample_item)
        INPUT = data.down_sample_item(selected_in).reshape(num_trials, 784)
        st.session_state.agent.next_state_batch(INPUT, selected_z, unsequenced=True)
        print("now training")
        st.session_state.agent._update_neuron_data(unsequenced=True)
        print("neurons updated")

        return

    for t in np.arange(num_trials):
        INPUT = data.down_sample(selected_in[t, :, :]).reshape(784)
        # print(INPUT)
        LABEL = selected_z[t]
        if is_training:
            user_STEPS = 1
            # print(INPUT)
            run_agent(user_STEPS, INPUT, LABEL)
            print("Trained on " + str(t))
        else:
            response_agent = run_agent(user_STEPS, INPUT, LABEL=[])
            if np.array_equal(response_agent, LABEL):
                correct_responses += 1
            print("Tested on " + str(t))
            print("TOTAL CORRECT-----------------" + str(correct_responses))

    trial_result = (correct_responses / num_trials) * 100
    st.session_state.correct_responses = correct_responses
    st.session_state.trial_result = trial_result
    print("Correct on {x}%".format(x=trial_result))


def run_canvas():
    input = data.down_sample(st.session_state.canvas_image).reshape(784)
    label = []
    user_steps = 10
    if st.session_state.train_canvas:
        label = list(np.binary_repr(int(canvas_label), 4))
        user_steps = 1
    response = run_agent(user_steps, input, LABEL=label)
    print(response)
    response_int = int("".join(str(x) for x in response), 2)
    st.session_state.canvas_int = response_int


def arr_to_img(img_array, enlarge_factor=15):
    # Convert the binary array to a numpy array
    img_array = np.array(img_array, dtype=np.uint8)

    # Scale the values to 0 or 255 (black or white)
    img_array = img_array * 255

    enlarged_array = np.repeat(img_array, enlarge_factor, axis=0)
    try:
        enlarged_array = np.repeat(enlarged_array, enlarge_factor, axis=1)
    except:
        enlarged_array = np.tile(enlarged_array, [enlarge_factor, 1])
        pass

    # Create an image from the array
    img = Image.fromarray(enlarged_array, mode="L")  # 'L' mode is for grayscale

    return img
    # # Save the image
    # img.save('binary_image.png')

    # If you want to display the image
    st.image(img)


# Basic streamlit setup
st.set_page_config(
    page_title="AO Labs Demo App",
    page_icon="https://i.imgur.com/j3jalQE.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:ali@aolabs.ai",
        "Report a bug": "mailto:ali@aolabs.ai",
        "About": "This is a demo of our AI features. Check out www.aolabs.ai and www.docs.aolabs.ai for more. Thank you!",
    },
)

streamlit_setup()

st.title("Understanding WNNs through MNIST")
st.write("### *a benchmark & demo by [aolabs.ai](https://www.aolabs.ai/)*")

train_max = 20000
test_max = 10000

agent_col, state_col = st.columns(2)

with agent_col:
    with st.expander("#### Batch Training", expanded=True):
        st.write("##### Training")

        st.write(
            "The agent can be trained on MNIST or standard fonts. When training on standard fonts, it trains on all of the digits from that font."
        )
        training_set_options = list(data.FONTS.keys())
        training_set_options.insert(0, "MNIST")
        st.session_state.training_sets = st.multiselect(
            "Select Training Sets:", options=training_set_options, default=("MNIST")
        )

        train_count = st.number_input(
            "Select number of randomly selected MNIST training samples",
            0,
            train_max,
            value=0,
        )
        st.button(
            "Train Agent",
            on_click=run_trials,
            args=(True, train_count, 1),
            disabled=len(st.session_state.training_sets) == 0,
        )

        st.write("##### Testing")
        st.write(
            "The agent can take multiple steps for each inference/prediction. Each step will result in an update of the agent's internal state. Modifying the steps per inference does not apply to training, at least not in this webapp."
        )
        t_count, t_steps = st.columns(2)
        with t_count:
            test_count = st.number_input(
                "select number of randomly selected tests", 1, test_max, value=1
            )
        with t_steps:
            user_STEPS = st.number_input(
                "How many steps per inference:", 1, 30, value=1
            )
        st.button(
            "Test Agent", on_click=run_trials, args=(False, test_count, user_STEPS)
        )

        st.markdown("""---""")

        # display trial result
        if "trial_result" in st.session_state:
            st.write("##### Test Results")
            st.write(
                "The agent predicted {correct} out of {total} images correctly, an accuracy of:".format(
                    correct=st.session_state.correct_responses, total=test_count
                )
            )
            st.write("# {result}%".format(result=st.session_state.trial_result))

    with st.expander("#### Continuous Learning", expanded=True):
        st.write(
            "You can also test your agent's inferences or train it on custom inputs made on the canvas below"
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
            # print(resized_gs)
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


# need to shape the data
# df = pd.DataFrame(st.session_state.agent.story[0, :].reshape((28, 28)))
with state_col:
    st.write("#### Agent State History")
    instruction_md = """
    Since our agents fall under reinforcement learning, they maintain an internal state that changes from step to step. \n
    The internal state of the agent is displayed below. Each state that the agent has held is accessible through the <number input> field, with greater numbers representing more recent states. \n
    The input and output layers can be seen as direct parallels to traditonal inputs and outputs, the internal layer is where WNNs differ from what you're used to seeing. \n 
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
        help="most recent state is currently: {}".format(st.session_state.agent.state),
    )

    I_col, Q_col, Z_col = st.columns(3)

    with I_col:
        st.write("##### Input")
        i_arr = st.session_state.agent.story[
            sel_state, st.session_state.agent.arch.I__flat
        ]
        i_arr = np.reshape(i_arr, [28, 28])
        i_img = arr_to_img(i_arr)
        st.image(i_img)
        # formatted_I = ""
        # for i in range(28):
        #     formatted_I += (str(i_arr[i]) + "\n")
        # st.text(formatted_I)

    with Q_col:
        st.write("##### Inner State")
        q_arr = st.session_state.agent.story[
            sel_state, st.session_state.agent.arch.Q__flat
        ]
        q_arr = np.reshape(q_arr, [28, 28])
        q_img = arr_to_img(q_arr)
        st.image(q_img)
        # formatted_Q = ""
        # for i in range(28):
        #     formatted_Q += (str(q_arr[i]) + "\n")
        # st.text(formatted_Q)

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
        # print(np_gs)

    # st.image(resized_gs)
    # input_image_gs_np = np.asarray(input_image_gs.getdata()).reshape(280, 280)
