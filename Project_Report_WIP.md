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

**refer to specific code modules, show implementation understanding**

## Project Overview 


The project models entail customized implmentations of the Google AI Magenta models for melody generating using a LSTM Recurrent Neural Network (Google AI Magenta, 2019c) and a Reinforcement Learning algorithm Tuning this network using Deep Q-Learning (Google AI Magenta, 2019d; Jaques et. al., 2017). Dataset choice, Model choice parameter options and customizations are informed by the requirements of the application context, application music practitioner literature, the authors own domain knowledge from 18 years as a music practitioner. 

## Project Goals

### Application Goals

The Goal is to design a music generator that is capable of generating melodies that are both musically expressive 
and have a certain coherency. The model will generate a monophonic melody line, (which will be responsive to a real time input melody. )

### Research Goals 

Thus, the project entails the implementation and evaluation of a neural network for a sequence modelling task in the context of music generation. The research goals are to explore, evaluate and compare different methods using sequence based machine learning and Artificial Intelligence for creative sequence generation in the field of musical melodies. 

(Franklin, 2006) - RNN useful in music gen. 

IT 1: Examine the effectivenss of a training LSTM RNN  on a general musical dataset of 11 133 musical examples. 

IT 2: Examine effeciveness of training LSTM RNN first on a larger general music dataset of ( ) (2.1), then on a smaller selection of context specific music examples (2.2) 

IT 3: Examine using Q-learning based on general musical rules to improve the musical melodies generated. 

(IT 4: Examine RL approaches to making the RNN learn the style of a human musician it plays along with) 

## Artificial Intelligence Research Context 

As with text and visual art generation, music generation falls within the space of computational creativity. 

In addition to providing a number of usecases, it is also interesting in the sense that creativity is a fundamental aspect of human intelligence (Reference). 

It is recognized that contemporary AI hold advantages over humans in analytical tasks such as identifying relationships and caputuring complex dynamics of large datasets. However, creativity is argued to be one of the areas in which humans still hold significant (Parry et. al, 2016 p.580; Jarrahi, 2018 p.582). Exploration of computational creativity is therefore an important step on the path to general or strong Artificial Intelligence (Russel & Norvig, 2016 p.16; Jaques et. al.,2017 ). 

"Music is an interesting test-bed for sequence generation, in that musical compositions adhere to a relatively well-defined set of structural rules. Any beginning music student learns that groups of notes belong to keys, chords follow progressions, and songs have consistent structures made up of musical phrases." (Jaques et. al., 2017). (Online) 

"The music theory rules implemented for the model are only a first attempt, and could easily be improved by someone with musical training."  (Jaques et. al., 2017). (Online) -  Improve music theory rule? 

Music generated my artificial intelligence    (AIVA, ;(game music)).


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

*Chords*: In monophonic (single melodies) these would often be "broken" in the sense that chord notes are played in sequence rather than simultaneously. 

*Harmonic Development/ Chord progression* :  meta patterns  for how these different types of relative value difference patterns alternate depending on their absolute values (harmonic development).

*Rhythm*: On/off sequences in time.  

**Performance** 

Phrasing

## Models


### LSTM Recurrent Neural Networks for sequence generation 

*Character RNN*  (Graves, 2013)

Goodfellow, I., Bengio, Y., & Courville., A.(2016 p.397)

Géron, A. (2017 p.407)

Magenta Melody RNN (reference)


### General Music Model

As with Language models, a music model can be first trained as a general model, then adapted to the specific application context. In Language this could entail building a general model for the language in question, then specialising the model to a certain application field using application specific data. In music, this training process could be utilised by first training the model on a general music data from which the model could learn general musical rules and conventions. The model could then be further trained on a data from specific style of music in order to specialise it to this field. 

In the first iteration, the model will consist of a music generation RNN LSTM. 

This is uses a similar basis as character level RNN´s developed for Natural Language Processing (Jaques et. al.,2017; Mikolov et. al.,2010) 

(Explain-). 


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


**General** 

A known failure mode of single step prediction RNN´s in sequence generation is the continuous repetition of the same token(Jaques et. al., 2017). The results of the LSTM RNN  were therefore evaluated on whether this failure mode was occuring. 


Another common failure mode of single step RNNs is the difficulty of ensuring coherent global structure. In music specifically, this global structure is built up by musical phrases (Jaques et. al., 2017). An evaluation criteria for the generated sequences is therefore to what extent they have a coherent global structure.


In a musical context this can refer to the following characteristics: 



**With primer** 

Capturing primer sequence pattern characteristics, i.e musical attributes. 


### General Model Evaluation


For the purely generative models, evaluation is based on how well the RNN LSTM model has been able to learn general musical parameters. Note sequences were generated based on several different primers, which aimed at 

*Primer: Starting note*

This examines what the AI generates freely when it does not try to match the musical characteristics of a specific melody. 

*Primer: Simple Major melody* 

This examines what the AI generates when it is fed a simple major melody, with two ascending steps and one step leading back ot the melody. It tests whether the melody is able:
- to generate melodies in the major scale
- imitate stepwise melody movement 
- use a leading note half step below the root note to lead back to the root note in melodic movements. 
- capture the steady melody rhytm with (...) as main sequence step length. 

*Primer: Simple Minor Melody.* 

This test is equivalent to the major melody, just in minor. It examines what the AI generates when it is fed a simple minor melody, with two ascending steps and one step leading back ot the melody. It tests whether the melody is able:
- to generate melodies in the minor scale
- imitate stepwise melody movement 
- use a leading note whole step below the root note to lead back to the root note in melodic movements. 

*Primer: Arpeggiated chord progression* 

This tests whether the model is able to capture arpeggiated chords, aka. individual chord notes in sequence, and chord progressions. 

*Primer: Bach Cello Suite Excerpt*

Thi tests whether the model is able to campture the melody style represented in the start of Bach´s cello suite. The style is characterised by  16th notes, which are short sequence step lengths, a combination of stepwise motion (delta of one or two half steps) and larger interval leaps (sequence step deltas between [6-9] ]. 

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


(### Preliminary Experiment: Bach Bot - Monophonic Cello RNN 

In order to familiarize myself with the Google Magenta toolkit and create a simple baseline model for further iterations, I first trained a simple model on using data from bachs cello suites. The advantage of this data is that it is mostly monophonic, e.g one melody, and captures many aspects of classical melodic and harmonic movement such as development of a repeated theme, development of a harmonic sequence and dramatic development. In data science terms, this means the data has a includes data patterns common in western musical traditions, such as absolute value differences in the sequence (melodic intervals), relative value difference patterns that are repeated with different absolute values locally (thematic development), and meta patterns  for how these different types of relative value difference patterns alternate depending on their absolute values (harmonic development). 

- Musical Analysis of bach cello suite in data science language

The initial cello suite was downloaded from an open source midi file repository (8 Notes, 2019). 

- Using default Magenta Tensorflow graphs )

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

**2.1: Complete Lakh and Maestro datasets

For the second iteration, a sixth primer melody was added. This was a simple phrase which would be similar to what would be used by the musical instrument player in the improvisation application context. 

Lakh dataset: 
Maestro:

It was a modal melody, e.g. a melody following a single 


**2.2: Style specific dataset**

*Training*

Transfer Learning (Géron, 2017 p.289)

The style specific data contained a much lower number of than the instances in the Lakh and Maestro training sets combined, 530 as opposed to 179 781 instances. The data was selected because it was closer to the instances likely encountered in the application context. 

Overfitting the model on this data as opposed to keeping it balanced between the general and specific datasets was therefore justified. (?)

A rather Naive approach was used for this second stage training, in which the training  steps was increased by a factor 179 781/530≈ 340 to give equal weight to the context data. (?)

(Paper on transfer learning & Two step training for language models) 


# Tuning the RNN with Deep Q Reinforcement Learning 

Implementation of the Magenta Deep Q Learning Music model created by Jaques et. al.(2017) using the authors own dataset (as described above) and customized musical theory rules in the reward function. 


### Deep Q Reinforcement Learning

Goodfellow et. al., 2016. 
Géron, 2017. 

## Deep Q Learning for Sequence Prediction 

### Magenta DQN Rl Tuner

Exploration mod: "egreedy" implementing epsilon greedy policy or boltzman to sample from its own outputs 
Priming mode: 

## 

## Customized Music Theory Rewards 

The reward_music_theory function defined in the RL Tuner model (Google AI, 2019d)  individual music theory subfunctions to compute a reward for desired musical outcomes. Among the default music theory rewards include playing a motif, repeating  a motif and following a certain scale. 

The original RL tuner model implemented musical theory rules based on some music domain knowledge and the musical treatise "A Practical Approach to Eighteenth-Century Counterpoint” by Robert Gauldin. In their paper and associated blog posts, Jaques et. al. (2017) encourage further exploration and customization of these music theory rules. 

For the present model these music theory rules were augmented to suit the specific application context of the model. 

Modal classical theory: 
Jazz theory: Pease & Pulling, 2001; Pease, 2004; Miller, 1996) 
Film music:  LOTR musical rules for flavour 


The original music theory rewards for the DQN RL tuner model only defined a C major scale. 
(https://github.com/tensorflow/magenta/blob/master/magenta/models/rl_tuner/rl_tuner_ops.py) 

Modal framework implementing relative scale pitches. These are based on the major modes of western music, which are used both in classical composition ( ) , Jazz Composition (Pease & Pulling, 2001; Pease, 2004; Miller, 1996) and cinematic music (Adams, 2010) 


# Conclusion 


# Directions for further research 



## Data sources
8 notes (2019).Bach - Cello Suite No.1 in G major, BWV 1007 (complete) midi file for Cello (midi). 8 Notes.  https://www.8notes.com/scores/14093.asp?ftype=midi [Accessed 18.04.2018]

http://www.midiworld.com/classic.htm#d

https://jazzomat.hfm-weimar.de/dbformat/dbcontent.html

http://www.acroche2.com/midi_jazz.html

https://groups.google.com/a/tensorflow.org/forum/#!topic/magenta-discuss/6ZLbzTjjpHM

## References
- Adams, D. (2010) *The Music of the Lord of The rings Films*
- Géron, A. (2017) *Hands-On Machine Learning with Scikit-Learn & Tensorflow* O´Reilly Media Inc, Sebastopol.
- Goldstein, G. (1982)  *Jazz Composer´s Companion* 
- Google AI (2019). Magenta - Make music and Art Using Machine LEarning. https://magenta.tensorflow.org/[Accessed 19.04.2018]
- Google AI Magenta (2019a). Magenta Github Repository. https://github.com/tensorflow/magenta [Accessed 19.04.2018]
- Google AI Magenta (2019b). The Maestro Dataset. *Tensorflow Magenta* https://magenta.tensorflow.org/datasets/maestro
- Google AI MAgenta (2019c). Melody RNN. *Magenta Github Repository* https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn 
- Google AI Magenta (2019d). RL Tuner. *Magenta Github Repository* https://github.com/tensorflow/magenta/tree/master/magenta/models/rl_tuner 
- Goodfellow, I., Bengio, Y., & Courville., A.(2016) *Deep Learning* MIT Press, London. 
- Graves, A (2013). Generating sequences with recurrent neural networks. *arXiv preprint:1308.0850.*
- Jaques, N 2016. Tuning Recurrent Neural Networks with Reinforcement Learning. *Google AI Magenta Blog* https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning  [Accessed 27.04.2019]
- Jaques, N., Gu,S., Turner, R E., & Eck, D. 2017.'TUNING RECURRENT NEURAL NETWORKS WITH REINFORCEMENT LEARNING'.  NIPS 2016 -Deep Reinforcement Learning Workshop. Available at: https://arxiv.org/pdf/1611.02796v2.pdf [Accessed 04.04.2019]
- Jarrahi, M.H (2018) Artificial intelligence and the future of work: Human-AI symbiosis in organizational decision making. Business Horizons. No. 61 pp. 577-586. 

- Mikolov et al.(2010) Recurrent neural network based language model. *In Interspeech*, volume 2, pp. 3.
- Miller, Ron. 1996. *Modal Jazz Composition & Harmony vol. 1.* Rottenberg: Advance Music
- Miller, Ron. 2000. *Model Jazz Composition & Harmony vol.2* Rottenber: Advance Music
- Parry, K. Cohen, M & Bhattacharya, S (2016) Rise of the Machines: A Critical Consideration of Automated Leadership Decision Making in Organisations. Group and Organisation Management. 2016 vol. Vol. 41(5) pp. 571–594
- Pease, T. & Pulling, K. 2001 *Modern Jazz Voicings* Berklee Press, Boston MA.
- Pease, T. 2004 *Jazz Composition: Theory and Practice* Berklee Press, Boston MA. 
- Russel, S & Norvig, P. (2016) *Artificial Intelligence - A Modern Approach*. 3rd Edition. Pearson Education Limited, Essex. 
-  https://arxiv.org/pdf/1810.12247.pdf
- https://arxiv.org/pdf/1803.05428.pdf
