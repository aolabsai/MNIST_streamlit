# MNIST Streamlit Demo
Maintainer: [spolisar](https://github.com/spolisar)

A streamlit app where users can train an agent to interpret drawn digits
This is done to show how you would approach a common machine learning task with Animo Omnis' core library as well as how that approach differs from the current status quo.


## Installation/Setup
### local environment
If you want to do this in a conda or venv environment set those up according to their instructions then do the following.

Install the requirements.
```shell
pip install -r requirements.txt
```

Install ao_core and ao_arch with the `pip install git+` method which lets you install python code from git repos.

```shell
pip install git+https://github.com/aolabsai/ao_arch git+https://github.com/aolabsai/ao_core
```

run it with the following command 
```shell
streamlit run main.py
```
The app will then be accessible at `localhost:8501`

### Docker
#### Build
This process assumes you already have an ssh key for github. If needed, follow [github's instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).
```shell
export DOCKER_BUILDKIT=1
eval `ssh-agent`
#if you have multiple password protected ssh keys you may want to specify your github key in the line below
ssh-add 
docker build -t "streamlit" --ssh default .
```
#### Run
```shell
docker run -p 8501:8501 streamlit
```
The app will then be accessible at `0.0.0.0:8501`

## Usage
Agents can be trained on data from multiple datasets with the largest being MNIST. The other datasets are based on existing fonts e.g. Arial. Agents can be trained on a mix of datasets, when doing that they train on every digit in a selected font and a random selection from MNIST. 

## How do these agents work?
Agents have 3 layers here, an input layer, state layer, and output layer. 

The input layer is where digit images are fed into the agent, before the agent sees the input we downsample it so grayscale values above 200 become 1 and below 200 become 0 then flatten the array. This gives us an input array of 784 0s and 1s.

The state layer is an representation of how the agent 'understands' its input.

The output layer is 4 binary digits representing the agent's prediction, which can easily be converted into an int.

Each of these layers are viewable on the right side of the streamlit app. Between running the agent on inputs, the agent is reset to a randomized state since identifying digits is not a continual process where the previous input should have an effect on the next input.

## Files Structure
arch_MNIST.py defines how the agent is structured.
mnist.pkl.gz is the MNIST training data 
Fonts/ has training data constructed from existing fonts such as Arial

## Future Work
- We may add the ability to download and upload trained agents in the future
- There are a couple cells in the Font training sets that seem to be causing errors without breaking anything, identifying and fixing those cells would be good.

## Contributing
Fork the repo make your changes and submit a pull request
