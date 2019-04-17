# Project Report: Music Generation with AI and Deep Learning (WIP) 


## Project Goal

The Goal is to design a music generator that is capable of generating melodies that are both musically expressive 
and have a certain coherency. The model will generate a monophonic melody line, which will be responsive to a real time input melody. 


## Application Context

The application context of the music project is in the realm of generating melodies for real time music improvisations in a cinematic style, such as film and game music. 


## Datasets

The data consisted of midi files and was selected with the overall application. 

### General melody generation

- General melody dataset 

### Style Specific melody generation 

Style specific dataset. 

As the data contained a much lower number of instances, this data was duplicated in order to provide specificity. 

Movie themes
http://www.midiworld.com/search/1/?q=movie%20themes

Movie themes of a symphonic nature were emphasized, (e.g in duplication). 

Video Game themes
http://www.midiworld.com/search/?q=video%20game%20themes

TV Themes
http://www.midiworld.com/search/?q=tv%20themes

Modal Classical 
Classical music from selected composers were also included as part of the style specific dataset. This was composers who were in the 
expressionist, impressionist and (..) paradigms as this was the moderns sounding style that would suit the model context. 

http://www.midiworld.com/classic.htm#d
http://www.midiworld.com/classic.htm#m

Theoretically, this is similar to training approaches used in Natural Language processing for building language context specific models (Reference needed). 

- Transform to melodies from 
- Think about target and prediction

Midi files were inspected using the musical notation software Sibelius. 

Style extraction? 

## Data processing 

The midi data was converted to the NoteSequence protocol developed by Google Magenta (Reference).   


## Model 

In the first iteration, the model will consist of a music generation RNN LSTM. 


The second iteration will then tune this LSTM using the Deep Q-learning approach by Jaques et. al.(2017). 

## References (Will be expanded for proper citation)
Jaques, N., Gu,S., Turner, R E., & Eck, D. 2017.'TUNING RECURRENT NEURAL NETWORKS WITH REINFORCEMENT LEARNING'.  NIPS 2016 Deep Reinforcement Learning Workshop. Available at: https://arxiv.org/pdf/1611.02796v2.pdf [Accessed 04.04.2019]

-  https://arxiv.org/pdf/1810.12247.pdf
- https://arxiv.org/pdf/1803.05428.pdf
