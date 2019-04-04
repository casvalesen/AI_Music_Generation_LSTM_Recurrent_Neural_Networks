# ST449 Course Projects

- This project accounts for 80% of the final mark.
- The project report is due on the 30th of April 2019.  

# Project Overview: Music Generation with RNN

## Topic

I would like to do my project on music generation using Recurrent Neural Networks. 
Google´s Magenta team are currently working on many interesting approaches to this. The project will make use of the Magenta Resources, but will also be original in that it will be a new combination of approaches with a specific musical context in mind. 
https://magenta.tensorflow.org/

## Model 

The Goal is to design a music generator that is capable of generating melodies that are both musically expressive 
and have a certain coherency. An interesting approach to this is to first train an LSTM on a dataset consisting of composed melodies, then tune this generator using rule based Q-learning. 

See: https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning

A goal in either the first or second phase of the project is also that the music generator is going to also be somewhat responsive to another player. This aspect will draw inspiration from the work done by Google Creative Lab on AI Duet. 
https://magenta.tensorflow.org/2017/02/16/ai-duet

## Datasets 

Several Datasets are available for the task, amongst others the MEASTRO dataset create by Google Magenta
https://magenta.tensorflow.org/maestro-wave2midi2wave
