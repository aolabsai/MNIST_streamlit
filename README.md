## AO Reference Design #1
# MNIST Streamlit Demo
Maintainer: [spolisar](https://github.com/spolisar), shane@aolabs.ai

A streamlit app where users can train an weightless neural network agent to identify hand-drawn digits (0-9) from the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).

![example of MNIST digits from Wiki](misc/mnist_example_wiki.png)

According to the [world leaderboard](https://paperswithcode.com/sota/image-classification-on-mnist), the highest accuracy achieved on MNIST is 99.87% by a variant of convolutional neural networkss (CNNs or cov-nets), with ~1.5M parameters trained on 60k training pairs. That accuracy is impressive but comes with the same limitations as larger pre-trained neural models (GPT is ~200B+ parameters nowawdays) that still hallucinate and can't learn around their mistakes.

AO labs is building AI agents than can learn after training. 

Modern machine learning started on MNIST, so this is our MNIST benchmark. Our agents here are made up of 784+4 trainable parameters in the form of weightless neurons (so no backprop) and you can train them on orders of magnitude less data (try 'em with 60 training inputs) to get good results (~60% accuracy on 60 samples!). More importantly, you can continuously (re)train the agent-- so if it makes a few mistakes with a particular number, draw new examples for it to learn from. You can also train the agent on standard fonts (like Times New Roman, Comic Sans, etc) in addition to MNIST training pairs.

In arch__MNIST.py, you can view the agent's particular neural architecture and try your hand at how different configurations affect performance.


## Installation & Setup

You can run this app in a docker container (recommended) or directly on your local environment. You'll need to `pip install` from our private repo ao_core, which is currently in private beta-- say hi on our [discord](https://discord.com/invite/nHuJc4Y4n7) for access!


### Docker Installation

1) Generate a GitHub Personal Access Token to ao_core    
    Go to https://github.com/settings/tokens?type=beta

2) Clone this repo and create a `.env` file in your local clone where you'll add the PAT as follows:
    `ao_github_PAT=token_goes_here`
    No spaces! See `.env_example`.

3) In a Git Bash terminal, build and run the Dockerfile with these commands:
```shell
export DOCKER_BUILDKIT=1

docker build --secret id=env,src=.env -t "ao_app" .

docker run -p 8501:8501 streamlit
```
You're done! Access the app at `localhost:8501` in your browser.

### Local Environment Installation

To install in a local conda or venv environment and run the app, use these commands:

```shell
pip install -r requirements.txt

streamlit run main.py
```
*Important:* You'll first need to uncomment lines 4 & 5 in the requirements.txt file.

You're done! Access the app at `localhost:8501` in your browser.


## How Do These Agents Work?
Agents are weightless neural network state machines made up of 3 layers, an input layer, an inner state, and output state. 

The input layer takes in the 28x28 pixel MNIST images, downsampled to B&W, as an an input array of 784 binary digits.

The inner state layer is a representation of how the agent 'understands' its input.

The output layer is 4 binary digits representing the agent's prediction, which is converted into an integer to match the MNIST labels.

Each of these layers are viewable on the right side of the streamlit app. Between running the agent on inputs, the agent is reset to a randomized state since identifying digits is not a continual process where the previous input should have an effect on the next input.

## File Structure
- arch__MNIST.py - defines how the agent's neural architecture (how many neurons and how they're connected)
- mnist.pkl.gz - the MNIST training data (60k training and 10k testing pairs)
- Fonts/ - has training data constructed from existing fonts such as Arial, Times New Roman, etc.

## Future Work
- We may add the ability to download and upload trained agents in the future
- There are a couple cells in the Font training sets that seem to be causing errors without breaking anything, identifying and fixing those cells would be good

## Contributing
Fork the repo, make your changes and submit a pull request. Join our [discord](https://discord.com/invite/nHuJc4Y4n7) and say hi!