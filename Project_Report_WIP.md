# Project Report: Music Generation with AI and Deep Learning (WIP) 

### Project Criteria: 

Individual project on a topic falling in the scope of the course
• Focus on methodological aspects, principles, and implementation

Allowing for
• Accounting for individual preferences
• Allowing to explore a given topic in more depth
• Allowing to gain hands-on experience in coding a solution

Selected:
• Implement and evaluate a neural network for a sequence modelling task

## Project Goals

### Application Goals

The Goal is to design a music generator that is capable of generating melodies that are both musically expressive 
and have a certain coherency. The model will generate a monophonic melody line, (which will be responsive to a real time input melody. )

### Research Goals 

Thus, the project entails the implementation and evaluation of a neural network for a sequence modelling task in the context of music generation. The research goals are to explore, evaluate and compare different methods using sequence based machine learning and Artificial Intelligence for creative sequence generation in the field of musical melodies. 

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

## Musical Data & Patterns

The data used for the model consists of musical material from the western musical traditions.  Although developing through history, the fundamental building blocks of this music have persisted from Bach (18th Century) to contemporary popular music (music history reference). A selection of these are: 

(check out music informatics for references here)

**Intervals**  : such as absolute value differences in the sequence (melodic intervals)

*Scales* : 

*Modal* (Pease & Pulling, 2001) (Pease, T. 2004)

**Melody** : A sequence of 

*Themes* : 

*Melodic Development* : Meta pattern     :relative value difference patterns that are repeated with different absolute values locally (thematic development) 

**Harmony** :

*Chords*:      In monophonic (single melodies) these would often be "broken" in the sense that chord notes are played in sequence rather than simultaneously. 

*Harmonic Development/ Chord progression* :  meta patterns  for how these different types of relative value difference patterns alternate depending on their absolute values (harmonic development).

*Rhythm*: On/off sequences in time.  

**Performance** 

Phrasing

## Models

### LSTM Recurrent Neural Networks for sequence generation 

Goodfellow, I., Bengio, Y., & Courville., A.(2016 p.397)

Géron, A. (2017 p.407)

Magenta Melody RNN (reference)


### General Music Model

As with Language models, a music model can be first trained as a general model, then adapted to the specific application context. In Language this could entail building a general model for the language in question, then specialising the model to a certain application field using application specific data. In music, this training process could be utilised by first training the model on a general music data from which the model could learn general musical rules and conventions. The model could then be further trained on a data from specific style of music in order to specialise it to this field. 

In the first iteration, the model will consist of a music generation RNN LSTM. 
This is uses a similar basis as character level RNN´s developed for Natural Language Processing (Jaques et. al.,2017) 


### Style Specific Model 

The second iteration will then tune this LSTM using the Deep Q-learning approach by Jaques et. al.(2017). 

## Evalation mechanisms 

As a creative RNN is in its nature not trying to predict an accurate value such as a time series or classification, 
evaluation of creative RNN is in its nature 

Although accuracy rate might give an objective measure of how well the model predicts a melody note, it is not very informative on how the model performs in context specific terms, namely in how it manages to be musically expressive and use established musical conventions. 

- It was therefore needed to define 

- Approaches to evaluation taken in papers, with refs (.....). Magenta refs, Jaques et. al., other papers for magenta.

This model evaluation definitions are in the field of explainable AI( ref). Since the objective measures or parameters captured by the model are not necessarily informative in the application context in the same way coefficients would be in regression, it is necessary to probe the AI model to generate outputs to get a general sense of how the model is performing and reasoning in the subject specific context (ref). 

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
- Building custom dataset

### General melody generation

- General melody dataset: Lakh dataset 178 579 midi files containing  

- Performance: Maestro, 1 202 midi files capturing world class performers classical musical material and performance 


### Style Specific melody generation 

Style specific dataset. 530 files combining both polyphonic and monophonic instances. 


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

**Jazz**

The purpose of including the jazz solos was for the model to learn jazz improvisational phrasing and performance. 

*Solos*
A selection of 64 jazz solos where also included in the dataset. 
Miles: Phrasing and clear motif statements. Both modal and fusion included for a more modern flavous(ref)
Coltrane: Phrasing and modal work     (ref)
Herbie Hancock: Phrasing and modal work (ref)
Wayne Shorter:   Phrasing and modal work  (ref) 
Kenny Wheeler: 70´s modal and ambiuous tonalities (ref)

https://jazzomat.hfm-weimar.de/dbformat/dbcontent.html

*Ensemble* 


## Data processing 

### Handling of Musical Data

- Midi format: Basic properties of musical data
Like text data, musical data in fundamentally sequential. 
Pitch at timestep 
Duration interpreted as on/off at timestep 

- The midi data was converted to the NoteSequence protocol developed by Google Magenta (Reference).   

(Graph of bach cello suite). 

###  Working with tf Records

- Refer to command line code for data processing 


### Preliminary Experiment: Bach Bot - Monophonic Cello RNN 

In order to familiarize myself with the Google Magenta toolkit and create a simple baseline model for further iterations, I first trained a simple model on using data from bachs cello suites. The advantage of this data is that it is mostly monophonic, e.g one melody, and captures many aspects of classical melodic and harmonic movement such as development of a repeated theme, development of a harmonic sequence and dramatic development. In data science terms, this means the data has a includes data patterns common in western musical traditions, such as absolute value differences in the sequence (melodic intervals), relative value difference patterns that are repeated with different absolute values locally (thematic development), and meta patterns  for how these different types of relative value difference patterns alternate depending on their absolute values (harmonic development). 

- Musical Analysis of bach cello suite in data science language

The initial cello suite was downloaded from an open source midi file repository (8 Notes, 2019). 

- Using default Magenta Tensorflow graphs 

## Iteration 1: Mono_rnn based on Magenta´s RNN LSTM Model 

- First part of Lakh dataset used, ... entries. 

- input data: 130 , 128 pitches on/off, 2 values for ....?

- Polyphonic midi files, aka. multiple simultaneous. Preprocessing extracted individual melodies / melody segments. 

- Mono_rnn
- RNN lstm model, 

**Model Evaluation**

Key? 
intervals? 
harmonic development? 

*Primer: One note* 



*Primer: Simple major Melody* 



*Primer: Simple minor Melody*



*Primer: Arpeggiated chord progression*



*Primer: Bach Cello Suite 4 bars 


## Iteration 2: Mono_rnn based on Magenta´s RNN LSTM Model, complete Lakh dataset (2.1) and style specific dataset (2.2)

For the second iteration, a sixth primer melody was added. This was a simple phrase which would be similar to what would be used by the musical instrument player in the improvisation application context. 

Lakh dataset: 
Maestro:

It was a modal melody, e.g. a melody following a single 


**2.1: Style specific dataset**

*Training*

Transfer Learning (Géron, 2017 p.289)

The style specific data contained a much lower number of than the instances in the Lakh and Maestro training sets combined, 530 as opposed to 179 781 instances. The data was selected because it was closer to the instances likely encountered in the application context. 

Overfitting the model on this data as opposed to keeping it balanced between the general and specific datasets was therefore justified. (?)

A rather Naive approach was used for this second stage training, in which the training  steps was increased by a factor 179 781/530≈ 340 to give equal weight to the context data. (?)

(Two step training for language models) 



# Conclusion 


# Directions for further research 



## Data sources
8 notes (2019).Bach - Cello Suite No.1 in G major, BWV 1007 (complete) midi file for Cello (midi). 8 Notes.  https://www.8notes.com/scores/14093.asp?ftype=midi [Accessed 18.04.2018]

http://www.midiworld.com/classic.htm#d

https://jazzomat.hfm-weimar.de/dbformat/dbcontent.html

http://www.acroche2.com/midi_jazz.html

https://groups.google.com/a/tensorflow.org/forum/#!topic/magenta-discuss/6ZLbzTjjpHM

## References (Will be expanded for proper citation)
- Géron, A. (2017) *Hands-On Machine Learning with Scikit-Learn & Tensorflow* O´Reilly Media Inc, Sebastopol. 
- Goodfellow, I., Bengio, Y., & Courville., A.(2016) *Deep Learning* MIT Press, London. 
- Jaques, N., Gu,S., Turner, R E., & Eck, D. 2017.'TUNING RECURRENT NEURAL NETWORKS WITH REINFORCEMENT LEARNING'.  NIPS 2016 -Deep Reinforcement Learning Workshop. Available at: https://arxiv.org/pdf/1611.02796v2.pdf [Accessed 04.04.2019]
- Jarrahi, M.H (2018) Artificial intelligence and the future of work: Human-AI symbiosis in organizational decision making. Business Horizons. No. 61 pp. 577-586. 
- Google AI (2019). Magenta - Make music and Art Using Machine LEarning. https://magenta.tensorflow.org/[Accessed 19.04.2018]
- Google AI Magenta (2019a). Magenta Github Repository. https://github.com/tensorflow/magenta [Accessed 19.04.2018]
- Google AI Magenta (2019b). The Maestro Dataset. *Tensorflow Magenta* https://magenta.tensorflow.org/datasets/maestro
- Parry, K. Cohen, M & Bhattacharya, S (2016) Rise of the Machines: A Critical Consideration of Automated Leadership Decision Making in Organisations. Group and Organisation Management. 2016 vol. Vol. 41(5) pp. 571–594
- Pease, T. & Pulling, K. 2001 *Modern Jazz Voicings* Berklee Press, Boston MA.
- Pease, T. 2004 *Jazz Composition: Theory and Practice* Berklee Press, Boston MA. 
- Russel, S & Norvig, P. (2016) Artificial Intelligence - A Modern Approach. 3rd Edition. Pearson Education Limited, Essex. 
-  https://arxiv.org/pdf/1810.12247.pdf
- https://arxiv.org/pdf/1803.05428.pdf
