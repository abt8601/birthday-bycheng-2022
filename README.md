# Birthday Message for Bo-Yu Cheng in 2022

This is the message I sent to [Bo-Yu Cheng](https://github.com/Nemo1999)
to congratulate his birthday in 2022.
This message takes the form of a puzzle, where I sent him part of the repository
(in particular [`main.py`](main.py) and [`out/model.pth`](out/model.pth))
and he has to find the hidden message ([`out/out.png`](out/out.png)).

This repository uses [Git LFS](https://git-lfs.github.com/)
to store the training data and the outputs.

## Background

**Warning:** This section contains spoilers.

People often congratulate each others' birthdays on Facebook
by posting messages like "Happy birthday" on their timelines.
For me, however, instead of throwing another "Happy birthday"
into an already large pool of almost identical messages,
I would like to deliver something different.
Therefore, I decided to put my creativity to the test and craft puzzles,
solving which would reveal my birthday messages.

I usually select the theme of the puzzle based on the recipient's background,
and in this case, it comes from his research focus.
Before his birthday,
we talked about his research topic for his master's in computer science.
He was working in computer vision
and was doing something related to neural implicit representation and NeRF.
As a person who hadn't entered the rabbit hole of machine learning,
I actually don't fully understand what he was talking about.
However, to me, a part of it sounded a lot like something
from the field of evolutionary computation that I had seen before: CPPN.
Therefore, for this year's puzzle,
I decided to use my limited understanding of CPPN and his research focus
and do the following:
train a neural network that encodes [an image of a cake](train-data/cake.png),
give him the code for training and the model weight,
and ask him to recover the image from the model.

In the same vein as using a CPPN to encode an image,
I used the pixel coordinates as the input
and the RGB values of the pixel as the output of the neural network.
However, instead of using a genetic algorithm to evolve the network
as people would normally do,
I designed the network architecture myself
and trained the network using a traditional gradient-based method,
which should be more in line with his research.
The network architecture I designed
was just a straightforward multi-layer perceptron,
with the number of hidden layers and the channel widths adjusted
based on the trade-off between performance and the memory pressure of training.
To me, the whole process was kind of like reviewing deep learning 101.
After a more-or-less satisfactory run,
I uploaded my puzzle ([`main.py`](main.py) and [`out/model.pth`](out/model.pth))
to an online file-sharing service,
posted the Base64-encoded sharing link to his timeline,
and asked him to "decrypt" it.
(Base64 encoding was necessary
to bypass Facebook not allowing me to post URLs for some reason.)

After receiving the puzzle and setting up his machine,
he quickly solved the puzzle and obtained the image.
Interestingly, the Base64 encoding gave him the most trouble.
He also quickly noticed some deficiencies in my model
and pointed me towards some recent works in the field.
In particular, one of them caught my greatest amount of attention:
using Fourier features allows an MLP to learn high-frequency functions,
which would otherwise bias towards low-frequency ones.
This overturned my perception
that pattern engineering is entirely unnecessary in a deep learning workflow.
Overall, he seemed to like the puzzle a lot,
and I had a great time creating this puzzle and exchanging ideas with him.
It also helped me deepen my understanding of deep learning.

## Files

- [`environment.yml`](environment.yml): The Anaconda environment.
- [`main.py`](main.py): The code that trains the neural network.
- [`train-data/`](train-data/): Training data.
  - [`cake.png`](train-data/cake.png): The training data; an image of a cake.
  - [`cake.xcf`](train-data/cake.xcf): The GIMP file of the training data.
- [`out/`](out/): Training outputs.
  - [`log.txt`](out/log.txt): Training log; the output of [`main.py`](main.py).
  - [`model.pth`](out/model.pth): The model weights.
  - `model-nnn.pth`: The model weights after training for `nnn` epochs.
  - [`out.png`](out/out.png): The output of the model.
  - `out-nnn.pth`: The output of the model after training for `nnn` epochs.

## Acknowledgements

The cake image in the training data comes from a glyph in the Symbola font.

## Licences

The source code is licensed under [the Unlicense](LICENSE).
The training data and the outputs (logs, model weights, and output images)
are licensed under the
[Creative Commons Zero v1.0 Universal license](https://creativecommons.org/publicdomain/zero/1.0/).
