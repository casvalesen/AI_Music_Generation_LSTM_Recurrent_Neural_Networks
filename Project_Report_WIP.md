# Project Report: Music Generation with AI and Deep Learning (WIP) 


## Project Goals

### Application Goals

The Goal is to design a music generator that is capable of generating melodies that are both musically expressive 
and have a certain coherency. The model will generate a monophonic melody line, (which will be responsive to a real time input melody. )

### Research Goals 

Explore, evaluate and compare different methods using sequence based machine learning and Artificial Intelligence for creative sequence generation in the field of musical melodies. 

IT 1: Examine the effectivenss of a training LSTM RNN  on a general musical dataset of 11 133 musical 

IT 2: Examine effeciveness of training LSTM RNN first on a larger general music dataset of () (2.1), then on a smaller selection of context specific music examples. 

IT 3: Examine using Q-learning based on general musical rules to improve the musical melodies generated. 

(IT 4: Examine RL approaches to making the RNN learn the style of a human musician it plays along with) 

## Artificial Intelligence Research Context 

As with text and visual art generation, music generation falls within the space of computational creativity. 

In addition to providing a number of usecases, it is also interesting in the sense that creativity is a fundamental aspect of human intelligence (Reference). 

It is recognized that contemporary AI hold advantages over humans in analytical tasks such as identifying relationships and caputuring complex dynamics of large datasets. However, creativity is argued to be one of the areas in which humans still hold significant (Parry et. al, 2016 p.580; Jarrahi, 2018 p.582). Exploration of computational creativity is therefore an important step on the path to general or strong Artificial Intelligence (Russel & Norvig, 2016 p.16; Jaques et. al.,2017 ). 

## Application Context

The application context of the music project is in the realm of generating melodies for real time music improvisations in a cinematic style, such as film and game music. 

## Models

### LSTM Recurrent Neural Networks for sequence generation 

Goodfellow, I., Bengio, Y., & Courville., A.(2016 p.397)

Géron, A. (2017 p.407)


### General Music Model

As with Language models, a music model can be first trained as a general model, then adapted to the specific application context. In Language this could entail building a general model for the language in question, then specialising the model to a certain application field using application specific data. In music, this training process could be utilised by first training the model on a general music data from which the model could learn general musical rules and conventions. The model could then be further trained on a data from specific style of music in order to specialise it to this field. 

In the first iteration, the model will consist of a music generation RNN LSTM. 
This is uses a similar basis as character level RNN´s developed for Natural Language Processing (Jaques et. al.,2017) 


### Style Specific Model 

The second iteration will then tune this LSTM using the Deep Q-learning approach by Jaques et. al.(2017). 

## Evalation mechanisms 

As a creative RNN is in its nature not trying to predict an accurate value such as a time series or classification, 
evaluation of creative RNN is in its nature 

- Approaches to evaluation taken in papers, with refs (.....). Magenta refs, Jaques et. al., other papers for magenta.

- Note on explainable AI. 

### General Model Evaluation

For the purely generative models, evaluation is based on how well the RNN LSTM model has been able to learn general musical parameters. Note sequences were generated based on several different primers, which aimed at 

*Primer: Starting note*

This examines what the AI generates freely when it does not try to match the musical characteristics of a specific melody. 

*Primer: Simple Major melody* 

This examines what the AI generates when it is fed a simple major melody, with two ascending steps and one step leading back ot the melody. It tests whether the melody is able:
- to generate melodies in the major scale
- imitate stepwise melody movement 
- use a leading note half step below the root note to lead back to the root note in melodic movements. 

*Primer: Simple Minor Melody.* 

This test is equivalent to the major melody, just in minor. It examines what the AI generates when it is fed a simple minor melody, with two ascending steps and one step leading back ot the melody. It tests whether the melody is able:
- to generate melodies in the minor scale
- imitate stepwise melody movement 
- use a leading note whole step below the root note to lead back to the root note in melodic movements. 

*Primer: Arpeggiated chord progression* 

This tests whether the model is able to capture arpeggiated chords, aka. individual chord notes in sequence, and chord progressions. 

## Datasets

The data consisted of midi files and was selected with the overall application. 

### General melody generation

- General melody dataset: Lakh dataset


### Style Specific melody generation 

Style specific dataset. 

As the data contained a much lower number of instances, this data was duplicated in order to provide specificity. 

**Movie themes**

http://www.midiworld.com/search/1/?q=movie%20themes

Movie themes of a symphonic nature were emphasized, (e.g in duplication). 

**Video Game themes**

http://www.midiworld.com/search/?q=video%20game%20themes

**TV Themes**

http://www.midiworld.com/search/?q=tv%20themes

**Modal Classical** 

Classical music from selected composers were also included as part of the style specific dataset. This was composers who were in the 
expressionist, impressionist and nasjonal romantic with inspiration of Scandinavian folk music (..) paradigms as this was the moderns sounding style that would suit the model context. 

http://www.midiworld.com/classic.htm#d
http://www.midiworld.com/classic.htm#m

Theoretically, this is similar to training approaches used in Natural Language processing for building language context specific models (Reference needed). 

- Transform to melodies from 
- Think about target and prediction

Midi files were inspected using the musical notation software Sibelius. 

Style extraction? 

**Jazz Solos**

The purpose of including the jazz solos was to 

A selection of jazz solos where also included in the dataset.
Miles: Phrasing and clear motif statements. Both modal and fusion included for a more modern flavous(ref)
Coltrane: Phrasing and modal work     (ref)
Herbie Hancock: Phrasing and modal work (ref)
Wayne Shorter:   Phrasing and modal work  (ref) 
Kenny Wheeler: 70´s modal and ambiuous tonalities (ref)



## Data processing 

### Handling of Musical Data

- Midi format: Basic properties of musical data
Like text data, musical data in fundamentally sequential. 
Pitch at timestep 
Duration interpreted as on/off at timestep 

- The midi data was converted to the NoteSequence protocol developed by Google Magenta (Reference).   

(Graph of bach cello suite). 

###  Working with tf Records


### Preliminary Experiment: Bach Bot - Monophonic Cello RNN 

In order to familiarize myself with the Google Magenta toolkit and create a simple baseline model for further iterations, I first trained a simple model on using data from bachs cello suites. The advantage of this data is that it is mostly monophonic, e.g one melody, and captures many aspects of classical melodic and harmonic movement such as development of a repeated theme, development of a harmonic sequence and dramatic development. In data science terms, this means the data has a includes data patterns common in western musical traditions, such as absolute value differences in the sequence (melodic intervals), relative value difference patterns that are repeated with different absolute values locally (thematic development), and meta patterns  for how these different types of relative value difference patterns alternate depending on their absolute values (harmonic development). 

- Musical Analysis of bach cello suite in data science language

The initial cello suite was downloaded from an open source midi file repository (8 Notes, 2019). 


- Using default Magenta Tensorflow graphs 

## Iteration 1: Mono_rnn based on Magenta´s RNN Model 

- First part of Lakh dataset used, ... entries. 

- input data: 130 , 128 pitches on/off, 2 values for ....?

- Polyphonic midi files, aka. multiple simultaneous. Preprocessing extracted individual melodies / melody segments. 

- Mono_rnn
- RNN lstm model, 

**Model Evaluation**

*Primer: One note* 

*Primer: Simple major Melody* 

*Primer: Simple minor Melody*

*Primer: Arpeggiated chord progression*


## Data sources
8 notes (2019).Bach - Cello Suite No.1 in G major, BWV 1007 (complete) midi file for Cello (midi). 8 Notes.  https://www.8notes.com/scores/14093.asp?ftype=midi [Accessed 18.04.2018]

http://www.midiworld.com/classic.htm#d

https://jazzomat.hfm-weimar.de/dbformat/dbcontent.html

https://groups.google.com/a/tensorflow.org/forum/#!topic/magenta-discuss/6ZLbzTjjpHM

## References (Will be expanded for proper citation)
-  Géron, A. (2017) *Hands-On Machine Learning with Scikit-Learn & Tensorflow* O´Reilly Media Inc, Sebastopol. 
- Goodfellow, I., Bengio, Y., & Courville., A.(2016) *Deep Learning* MIT Press, London. 
- Jaques, N., Gu,S., Turner, R E., & Eck, D. 2017.'TUNING RECURRENT NEURAL NETWORKS WITH REINFORCEMENT LEARNING'.  NIPS 2016 -Deep Reinforcement Learning Workshop. Available at: https://arxiv.org/pdf/1611.02796v2.pdf [Accessed 04.04.2019]
- Jarrahi, M.H (2018) Artificial intelligence and the future of work: Human-AI symbiosis in organizational decision making. Business Horizons. No. 61 pp. 577-586. 
- Google AI (2019). Magenta - Make music and Art Using Machine LEarning. https://magenta.tensorflow.org/[Accessed 19.04.2018]
- Google AI Magenta (2019). Magenta Github Repository. https://github.com/tensorflow/magenta [Accessed 19.04.2018]
- Parry, K. Cohen, M & Bhattacharya, S (2016) Rise of the Machines: A Critical Consideration of Automated Leadership Decision Making in Organisations. Group and Organisation Management. 2016 vol. Vol. 41(5) pp. 571–594
- Russel, S & Norvig, P. (2016) Artificial Intelligence - A Modern Approach. 3rd Edition. Pearson Education Limited, Essex. 
-  https://arxiv.org/pdf/1810.12247.pdf
- https://arxiv.org/pdf/1803.05428.pdf
